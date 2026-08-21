// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackParameterHelpers.hpp"
#include "Acts/EventData/TrackStatePropMask.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/DafReferenceState.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"

#include <cstddef>
#include <vector>

namespace Acts::detail {

/// Kalman-filter a recorded DAF reference trajectory, without propagating.
///
/// This is the second half of GenFit's KalmanFitterRefTrack: the reference
/// pass (DafActor with recordReference set) has already produced, per surface,
/// the reference parameters, the transport Jacobian F, the process noise N,
/// the delta state c and the reference-calibrated candidates. All that is left
/// is the affine prediction
///
///     p_k|k-1 = F_k * p_k-1|k-1 + c_k
///     C_k|k-1 = F_k * C_k-1|k-1 * F_k^T + N_k
///
/// followed by the usual weighted combination and gain-matrix update. No
/// stepper, no navigator, no material lookup -- every hit was linearised on
/// one and the same trajectory, which is the entire point.
///
/// The filter starts at the first recorded state with the reference parameters
/// and the supplied start covariance, exactly as GenFit does ("Use reference
/// state and seed cov as start"): the deviation from the reference starts at
/// zero, while the covariance claims no information.
///
/// @tparam traj_t The trajectory type
/// @tparam track_container_t The track container type
///
/// @param reference The recorded reference trajectory, in propagation order
/// @param candidateStore Persistent store holding the calibrated candidates
///        that DafReferenceState::candidateIndices point into
/// @param startCovariance Covariance the filter starts with, i.e. GenFit's
///        blown-up seed covariance
/// @param candidateState Per-surface annealed weights for this pass
/// @param options The fitter options, for the extensions and contexts
/// @param trackContainer Track container to build the filtered track in
/// @param logger Logger for diagnostic output
///
/// @return The filtered track, ready to be smoothed
template <typename traj_t, typename track_container_t>
Result<typename track_container_t::TrackProxy> dafReferenceFilter(
    const DafReferenceTrajectory& reference, const traj_t& candidateStore,
    const BoundSquareMatrix& startCovariance,
    const DafCandidateMap& candidateState,
    const DafFitterOptions<traj_t>& options, track_container_t& trackContainer,
    const Logger& logger) {
  if (reference.empty()) {
    ACTS_DEBUG("Reference trajectory is empty");
    return DafFitterError::NoMeasurementFound;
  }

  auto& trajectory = trackContainer.trackStateContainer();

  std::size_t previousIndex = kTrackIndexInvalid;
  std::size_t lastMeasurementIndex = kTrackIndexInvalid;
  std::size_t measurementStates = 0;

  BoundVector previousFiltered = BoundVector::Zero();
  BoundSquareMatrix previousFilteredCovariance = BoundSquareMatrix::Zero();
  BoundVector previousReference = BoundVector::Zero();
  bool hasPrevious = false;

  for (const DafReferenceState& ref : reference) {
    const TrackStatePropMask mask =
        ref.hasMeasurement
            ? (TrackStatePropMask::Predicted | TrackStatePropMask::Filtered |
               TrackStatePropMask::Jacobian | TrackStatePropMask::Calibrated)
            : (TrackStatePropMask::Predicted | TrackStatePropMask::Jacobian);

    auto trackStateProxy = trajectory.makeTrackState(mask, previousIndex);
    trackStateProxy.setReferenceSurface(ref.surface->getSharedPtr());
    trackStateProxy.jacobian() = ref.jacobian;
    trackStateProxy.pathLength() = ref.pathLength;

    if (!hasPrevious) {
      // Start of the track: the estimate IS the reference, so the deviation is
      // zero by construction. The covariance is the start covariance TRANSPORTED
      // onto this surface -- J*C*J^T + N, with J and N recorded from the
      // propagation start. Using the start covariance unchanged would apply a
      // covariance defined on the seed surface (a perigee) to a straw surface,
      // where its loc0/loc1 block means something entirely different.
      trackStateProxy.predicted() = ref.parameters;
      trackStateProxy.predictedCovariance() =
          ref.jacobian * startCovariance * ref.jacobian.transpose() + ref.noise;
    } else {
      // p_k|k-1 = p_ref,k + F_k * (p_k-1|k-1 - p_ref,k-1)
      //
      // Algebraically identical to GenFit's p = F*p + c, since
      // c_k = p_ref,k - F_k*p_ref,k-1 -- but written in the deviation form
      // because ACTS bound parameters carry phi and theta, and the raw affine
      // form cannot survive the branch cut. If the filtered estimate and the
      // reference land on opposite sides of +-pi, their componentwise
      // difference is ~2*pi rather than ~0 and the prediction is destroyed.
      // GenFit never meets this: its 5D plane parameters (q/p, u', v', u, v)
      // have no periodic coordinate. subtract/addBoundParameters wrap both
      // angles into the symmetric range, which is exactly what the small
      // deviations this filter works with need.
      const BoundVector deviation =
          subtractBoundParameters(previousFiltered, previousReference);
      trackStateProxy.predicted() =
          addBoundParameters(ref.parameters, ref.jacobian * deviation);
      trackStateProxy.predictedCovariance() =
          ref.jacobian * previousFilteredCovariance * ref.jacobian.transpose() +
          ref.noise;
    }

    auto typeFlags = trackStateProxy.typeFlags();
    typeFlags.setHasParameters();
    if (ref.hasMaterial) {
      typeFlags.setHasMaterial();
    }
    if (ref.isHole) {
      typeFlags.setIsHole();
    }

    if (ref.hasMeasurement) {
      const auto candStateIt = candidateState.find(ref.surface);
      if (candStateIt == candidateState.end()) {
        ACTS_DEBUG("No candidate state for surface "
                   << ref.surface->geometryId());
        return DafFitterError::NoMeasurementFound;
      }
      const DafCandidateState& candState = candStateIt->second;
      if (candState.weights.size() != ref.candidateIndices.size()) {
        ACTS_DEBUG("Weight/candidate count mismatch on surface "
                   << ref.surface->geometryId() << ": "
                   << candState.weights.size() << " vs "
                   << ref.candidateIndices.size());
        return DafFitterError::NoMeasurementFound;
      }

      std::vector<typename traj_t::ConstTrackStateProxy> constCandidates;
      constCandidates.reserve(ref.candidateIndices.size());
      for (std::size_t index : ref.candidateIndices) {
        constCandidates.push_back(candidateStore.getTrackState(index));
      }

      options.extensions.weightedCombiner(constCandidates, candState.weights,
                                          trackStateProxy);

      // Same convention as DafActor: the state resolves to the currently
      // favoured candidate, which becomes exact once the weights sharpen.
      std::size_t bestIdx = 0;
      for (std::size_t i = 1; i < candState.weights.size(); ++i) {
        if (candState.weights[i] > candState.weights[bestIdx]) {
          bestIdx = i;
        }
      }
      if (bestIdx < candState.candidates.size()) {
        trackStateProxy.setUncalibratedSourceLink(
            SourceLink(candState.candidates[bestIdx]));
      }

      auto updateRes =
          options.extensions.updater(options.geoContext, trackStateProxy,
                                     logger);
      if (!updateRes.ok()) {
        ACTS_DEBUG("Update step failed on surface "
                   << ref.surface->geometryId() << ": " << updateRes.error());
        return updateRes.error();
      }
      typeFlags.setIsMeasurement();

      lastMeasurementIndex = trackStateProxy.index();
      ++measurementStates;
    } else {
      trackStateProxy.shareFrom(trackStateProxy, TrackStatePropMask::Predicted,
                                TrackStatePropMask::Filtered);
    }

    previousIndex = trackStateProxy.index();
    previousFiltered = trackStateProxy.filtered();
    previousFilteredCovariance = trackStateProxy.filteredCovariance();
    previousReference = ref.parameters;
    hasPrevious = true;
  }

  if (measurementStates == 0) {
    ACTS_DEBUG("Reference filter found no measurement states");
    return DafFitterError::NoMeasurementFound;
  }

  auto track = trackContainer.makeTrack();
  track.tipIndex() = lastMeasurementIndex;
  calculateTrackQuantities(track);

  return track;
}

}  // namespace Acts::detail

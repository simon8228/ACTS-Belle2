// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Common.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/MultiTrajectoryHelpers.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/detail/CorrectedTransformationFreeToBound.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace Acts {

/// Result payload returned by the Deterministic Annealing Filter's actor for
/// a single forward+smooth pass. Mirrors @ref KalmanFitterResult.
template <typename traj_t>
struct DafFitterResult {
  /// Fitted states that the actor has handled.
  traj_t* fittedStates{nullptr};

  /// Index of the last measurement state in the multitrajectory.
  std::size_t lastMeasurementIndex = kTrackIndexInvalid;

  /// Index of the last state in the multitrajectory.
  std::size_t lastTrackIndex = kTrackIndexInvalid;

  /// The optional Parameters at the provided surface
  std::optional<BoundTrackParameters> fittedParameters;

  /// Counter for states with a measurement group
  std::size_t measurementStates = 0;

  /// Counter for measurement holes
  std::size_t measurementHoles = 0;

  /// Counter for handled states
  std::size_t processedStates = 0;

  /// Indicator if track fitting has been done
  bool finished = false;

  /// Measurement surfaces without hits
  std::vector<const Surface*> missedActiveSurfaces;

  /// Path limit aborter
  PathLimitReached pathLimitReached;

  /// Step count as of the last time `processedStates` increased. Used by the
  /// stall guard to detect propagation that is making no net forward
  /// progress (e.g. an oscillating, non-converging intersection search for
  /// the next target surface -- observed for very low-momentum tracks whose
  /// helix radius is tiny relative to the distance to their next target)
  /// well before the step budget is exhausted.
  std::size_t stepAtLastProgress = 0;

  /// `processedStates` value as of the last stall check, to detect when it
  /// changes.
  std::size_t processedStatesAtLastProgress = 0;
};

namespace detail {

/// @brief Propagator Actor plugin for the Deterministic Annealing Filter
///
/// Forked from @ref KalmanFitter's Actor rather than parametrized, since
/// their `filter()` methods diverge structurally in the measurement branch
/// (single calibration+update vs. multi-candidate weighting+combination)
/// while sharing the hole/material-only branch and abort/finalization logic
/// verbatim.
template <typename traj_t>
class DafActor {
 public:
  /// Broadcast the result_type
  using result_type = DafFitterResult<traj_t>;

  /// The target surface aborter
  SurfaceReached targetReached{std::numeric_limits<double>::lowest()};

  /// Allows retrieving the competing candidate source links for a surface
  std::unordered_map<const Surface*, DafMeasurementGroup>
      inputMeasurementGroups;

  /// Non-owning pointer to the outer driver's persistent per-candidate
  /// annealing state, keyed the same way as inputMeasurementGroups. Read-only
  /// here: the driver populates each surface's prior weights (from the
  /// previous pass's smoothed trajectory, or uniform on the first pass)
  /// before propagation starts, and updates them itself after smoothing.
  DafCandidateMap* candidateState{nullptr};

  /// Scratch trajectory used to build per-candidate calibrated-only track
  /// states. Cleared before each surface's candidates are built and never
  /// attached to the result, so the real output trajectory is never bloated
  /// with N-1 discarded candidate rows per multi-candidate surface.
  traj_t* scratchStates{nullptr};

  /// Maximum number of propagation steps allowed without any forward
  /// progress (a new measurement, hole, or material-only surface being
  /// processed) before the pass is aborted as stalled. Guards against
  /// oscillating, non-converging intersection searches for the next target
  /// surface (observed for very low-momentum tracks whose helix radius has
  /// become tiny relative to the distance to their next target) burning
  /// through the full step budget instead of failing fast.
  std::size_t stallStepLimit = 2000;

  /// Whether to consider multiple scattering.
  bool multipleScattering = true;

  /// Whether to consider energy loss.
  bool energyLoss = true;

  /// Whether to include non-linear correction during global to local
  /// transformation
  FreeToBoundCorrection freeToBoundCorrection;

  /// Input MultiTrajectory
  std::shared_ptr<traj_t> outputStates;

  DafFitterExtensions<traj_t> extensions;

  /// Calibration context for the fit
  const CalibrationContext* calibrationContext{nullptr};

  /// End of world aborter
  EndOfWorldReached endOfWorldReached;

  /// Volume constraint aborter
  VolumeConstraintAborter volumeConstraintAborter;

  /// The logger instance
  const Logger* actorLogger{nullptr};

  /// Logger helper
  const Logger& logger() const { return *actorLogger; }

  /// @brief DAF actor operation
  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  Result<void> act(propagator_state_t& state, const stepper_t& stepper,
                   const navigator_t& navigator, result_type& result,
                   const Logger& /*logger*/) const {
    assert(result.fittedStates && "No MultiTrajectory set");

    if (result.finished) {
      return Result<void>::success();
    }

    ACTS_VERBOSE("DafActor step at pos: "
                 << stepper.position(state.stepping).transpose()
                 << " dir: " << stepper.direction(state.stepping).transpose()
                 << " momentum: "
                 << stepper.absoluteMomentum(state.stepping));

    if (result.pathLimitReached.internalLimit ==
        std::numeric_limits<double>::max()) {
      Acts::detail::setupLoopProtection(state, stepper,
                                        result.pathLimitReached, true,
                                        logger());
    }

    const Surface* surface = navigator.currentSurface(state.navigation);
    if (surface != nullptr) {
      ACTS_VERBOSE("Perform " << state.options.direction << " filter step");
      auto res = filter(*surface, state, stepper, navigator, result);
      if (!res.ok()) {
        ACTS_DEBUG("Error in " << state.options.direction
                               << " filter: " << res.error());
        return res.error();
      }
    }

    if (result.processedStates != result.processedStatesAtLastProgress) {
      result.processedStatesAtLastProgress = result.processedStates;
      result.stepAtLastProgress = state.steps;
    }
    if (state.steps - result.stepAtLastProgress > stallStepLimit) {
      const Vector3 pos = stepper.position(state.stepping);
      const Vector3 dir = stepper.direction(state.stepping);
      const double p = stepper.absoluteMomentum(state.stepping);
      const double pT = p * std::hypot(dir.x(), dir.y());
      const Surface* targetSurface = targetReached.surface;
      ACTS_DEBUG(
          "DafActor: no forward progress for "
          << stallStepLimit
          << " steps, aborting as stalled instead of exhausting the "
             "step budget. "
          << "steps=" << state.steps << " pathLength=" << state.pathLength
          << " pos=" << pos.transpose() << " dir=" << dir.transpose()
          << " p=" << p << " pT=" << pT
          << " charge=" << stepper.charge(state.stepping)
          << " processedStates=" << result.processedStates << "/"
          << inputMeasurementGroups.size()
          << " measurementStates=" << result.measurementStates
          << " currentSurface="
          << (surface != nullptr ? surface->geometryId().value() : 0)
          << " targetSurface="
          << (targetSurface != nullptr ? targetSurface->geometryId().value()
                                        : 0));
      return DafFitterError::PropagationStalled;
    }

    const bool isTrackComplete =
        result.measurementStates == inputMeasurementGroups.size();
    const bool isEndOfWorldReached =
        endOfWorldReached.checkAbort(state, stepper, navigator, logger());
    const bool isVolumeConstraintReached = volumeConstraintAborter.checkAbort(
        state, stepper, navigator, logger());
    const bool isPathLimitReached = result.pathLimitReached.checkAbort(
        state, stepper, navigator, logger());
    const bool isTargetReached =
        targetReached.checkAbort(state, stepper, navigator, logger());
    if (isTrackComplete || isEndOfWorldReached || isVolumeConstraintReached ||
        isPathLimitReached || isTargetReached) {
      ACTS_VERBOSE(
          "Finalizing DAF pass: "
          << (isTrackComplete ? "track complete; " : "")
          << (isEndOfWorldReached ? "end of world reached; " : "")
          << (isVolumeConstraintReached ? "volume constraint reached; " : "")
          << (isPathLimitReached ? "path limit reached; " : "")
          << (isTargetReached ? "target surface reached; " : ""));

      if (isTargetReached) {
        auto res = stepper.boundState(state.stepping, *targetReached.surface);
        if (!res.ok()) {
          ACTS_DEBUG("Error while acquiring bound state for target surface: "
                     << res.error() << " " << res.error().message());
          return res.error();
        } else {
          const auto& [boundParams, jacobian, pathLength] = *res;
          result.fittedParameters = boundParams;
        }
      }

      result.finished = true;
    }

    return Result<void>::success();
  }

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  bool checkAbort(propagator_state_t& /*state*/, const stepper_t& /*stepper*/,
                  const navigator_t& /*navigator*/, const result_type& result,
                  const Logger& /*logger*/) const {
    return result.finished;
  }

  /// @brief DAF actor operation: filter
  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  Result<void> filter(const Surface& surface, propagator_state_t& state,
                      const stepper_t& stepper, const navigator_t& navigator,
                      result_type& result) const {
    const bool precedingMeasurementExists = result.measurementStates > 0;
    const bool surfaceIsSensitive = surface.isSensitive();
    const bool surfaceHasMaterial = surface.surfaceMaterial() != nullptr;

    const auto groupIt = inputMeasurementGroups.find(&surface);
    if (groupIt != inputMeasurementGroups.end()) {
      const DafMeasurementGroup& group = groupIt->second;
      ACTS_VERBOSE("Measurement surface " << surface.geometryId()
                                          << " detected with "
                                          << group.size() << " candidate(s).");

      stepper.transportCovarianceToBound(state.stepping, surface,
                                         freeToBoundCorrection);

      Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::PreUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger());

      TrackStatePropMask mask =
          TrackStatePropMask::Predicted | TrackStatePropMask::Filtered |
          TrackStatePropMask::Jacobian | TrackStatePropMask::Calibrated;
      typename traj_t::TrackStateProxy trackStateProxy =
          result.fittedStates->makeTrackState(mask, result.lastTrackIndex);

      trackStateProxy.setReferenceSurface(surface.getSharedPtr());
      auto res = stepper.boundState(state.stepping, surface, false,
                                    freeToBoundCorrection);
      if (!res.ok()) {
        ACTS_DEBUG("Propagate to surface " << surface.geometryId()
                                           << " failed: " << res.error());
        return res.error();
      }
      const auto& [boundParams, jacobian, pathLength] = *res;

      trackStateProxy.predicted() = boundParams.parameters();
      trackStateProxy.predictedCovariance() = state.stepping.cov;
      trackStateProxy.jacobian() = jacobian;
      trackStateProxy.pathLength() = pathLength;

      assert(scratchStates != nullptr &&
             "DafActor requires a scratch trajectory");
      scratchStates->clear();

      std::vector<typename traj_t::TrackStateProxy> candidates;
      candidates.reserve(group.size());
      for (std::size_t i = 0; i < group.size(); ++i) {
        TrackStatePropMask candidateMask =
            (i == 0) ? (TrackStatePropMask::Predicted |
                       TrackStatePropMask::Jacobian |
                       TrackStatePropMask::Calibrated)
                    : TrackStatePropMask::Calibrated;
        auto candidate =
            scratchStates->makeTrackState(candidateMask, kTrackIndexInvalid);
        if (i == 0) {
          candidate.predicted() = trackStateProxy.predicted();
          candidate.predictedCovariance() = trackStateProxy.predictedCovariance();
          candidate.jacobian() = trackStateProxy.jacobian();
        } else {
          candidate.shareFrom(candidates.front(), TrackStatePropMask::Predicted);
          candidate.shareFrom(candidates.front(), TrackStatePropMask::Jacobian);
        }
        extensions.calibrator(state.geoContext, *calibrationContext, group[i],
                              candidate);
        candidates.push_back(candidate);
      }

      std::vector<typename traj_t::ConstTrackStateProxy> constCandidates(
          candidates.begin(), candidates.end());

      // Use this surface's current prior weights, computed by the outer
      // driver from the PREVIOUS pass's smoothed trajectory (uniform on the
      // very first pass). The actor deliberately does not compute weights
      // itself: doing so from this pass's own freshly-propagated predicted
      // state would make weights depend only on a single, not-yet-smoothed
      // estimate, which is far too weak to discriminate candidates near the
      // start of a track (the smoothed state pulls in information from
      // later hits too) and previously caused the fit to diverge.
      assert(candidateState != nullptr);
      auto& candState = (*candidateState)[&surface];
      assert(candState.weights.size() == group.size() &&
             "DafActor requires prior weights to be initialized by the "
             "outer driver before propagation");

      extensions.weightedCombiner(constCandidates, candState.weights,
                                  trackStateProxy);

      if (logger().doPrint(Logging::DEBUG)) {
        std::stringstream candSs;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
          candSs << " [w=" << candState.weights[i] << " meas="
                 << candidates[i].effectiveCalibrated().transpose() << "]";
        }
        const std::size_t combinedMeasdim = trackStateProxy.calibratedSize();
        ACTS_DEBUG("DafActor: combined "
                   << candidates.size() << " candidate(s) on surface "
                   << surface.geometryId() << " predicted="
                   << trackStateProxy.predicted().transpose()
                   << " candidates:" << candSs.str() << " -> combined meas="
                   << trackStateProxy.effectiveCalibrated().transpose()
                   << " combined covariance diag="
                   << trackStateProxy.effectiveCalibratedCovariance()
                          .diagonal()
                          .transpose()
                   << " measdim=" << combinedMeasdim);
      }

      // The real (non-scratch) track state otherwise has no uncalibrated
      // source link at all, since only the scratch candidates were
      // individually calibrated -- but downstream consumers (truth
      // matching, visualization, ...) expect every measurement track state
      // to resolve to one. Use the currently-favored candidate, i.e. the
      // one with the highest annealed weight; this is DAF's current best
      // guess of which raw measurement this state represents, and becomes
      // exact once weights sharpen to a hard assignment at low temperature.
      {
        std::size_t bestIdx = 0;
        for (std::size_t i = 1; i < candState.weights.size(); ++i) {
          if (candState.weights[i] > candState.weights[bestIdx]) {
            bestIdx = i;
          }
        }
        trackStateProxy.setUncalibratedSourceLink(SourceLink(group[bestIdx]));
      }

      auto typeFlags = trackStateProxy.typeFlags();
      typeFlags.setHasParameters();
      if (surfaceHasMaterial) {
        typeFlags.setHasMaterial();
      }

      auto updateRes =
          extensions.updater(state.geoContext, trackStateProxy, logger());
      if (!updateRes.ok()) {
        ACTS_DEBUG("Update step failed: " << updateRes.error());
        return updateRes.error();
      }
      typeFlags.setIsMeasurement();

      result.lastTrackIndex = trackStateProxy.index();

      if (logger().doPrint(Logging::DEBUG)) {
        const double predictedP =
            1. / std::abs(trackStateProxy.predicted()[eBoundQOverP]);
        const double predictedPt =
            predictedP * std::sin(trackStateProxy.predicted()[eBoundTheta]);
        const double filteredP =
            1. / std::abs(trackStateProxy.filtered()[eBoundQOverP]);
        const double filteredPt =
            filteredP * std::sin(trackStateProxy.filtered()[eBoundTheta]);
        ACTS_DEBUG("DafActor: Kalman update on surface "
                   << surface.geometryId() << " predicted p=" << predictedP
                   << " pT=" << predictedPt << " -> filtered p=" << filteredP
                   << " pT=" << filteredPt << " (ratio p="
                   << filteredP / predictedP << ")");
      }

      ACTS_VERBOSE("Filtering step successful, updated parameters are:\n"
                   << trackStateProxy.filtered().transpose());
      stepper.update(state.stepping,
                     MultiTrajectoryHelpers::freeFiltered(
                         state.options.geoContext, trackStateProxy),
                     trackStateProxy.filtered(),
                     trackStateProxy.filteredCovariance(), surface);
      ++result.measurementStates;

      Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::PostUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger());
      ++result.processedStates;
      result.measurementHoles = result.missedActiveSurfaces.size();
      result.lastMeasurementIndex = result.lastTrackIndex;

    } else if ((precedingMeasurementExists && surfaceIsSensitive) ||
               surfaceHasMaterial) {
      // Hole / material-only branch: no competing candidates involved,
      // identical to KalmanFitter::Actor::filter().
      TrackStatePropMask mask =
          TrackStatePropMask::Predicted | TrackStatePropMask::Jacobian;
      typename traj_t::TrackStateProxy trackStateProxy =
          result.fittedStates->makeTrackState(mask, result.lastTrackIndex);

      trackStateProxy.setReferenceSurface(surface.getSharedPtr());
      auto res = stepper.boundState(state.stepping, surface, true,
                                    freeToBoundCorrection);
      if (!res.ok()) {
        return res.error();
      }
      const auto& [boundParams, jacobian, pathLength] = *res;

      trackStateProxy.predicted() = boundParams.parameters();
      trackStateProxy.predictedCovariance() = state.stepping.cov;
      trackStateProxy.jacobian() = jacobian;
      trackStateProxy.pathLength() = pathLength;

      trackStateProxy.shareFrom(trackStateProxy, TrackStatePropMask::Predicted,
                                TrackStatePropMask::Filtered);

      auto typeFlags = trackStateProxy.typeFlags();
      typeFlags.setHasParameters();

      if (surfaceHasMaterial) {
        typeFlags.setHasMaterial();
      }

      if (surfaceIsSensitive && precedingMeasurementExists) {
        ACTS_VERBOSE("Detected hole on " << surface.geometryId());
        typeFlags.setIsHole();
      } else if (surfaceIsSensitive) {
        ACTS_VERBOSE("Skip hole (no preceding measurements) on surface "
                     << surface.geometryId());
      } else if (surfaceHasMaterial) {
        ACTS_VERBOSE("Detected in-sensitive surface " << surface.geometryId());
      }

      result.lastTrackIndex = trackStateProxy.index();

      if (trackStateProxy.typeFlags().isHole()) {
        result.missedActiveSurfaces.push_back(&surface);
      }

      ++result.processedStates;

      Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::FullUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger());
    }

    return Result<void>::success();
  }
};

}  // namespace detail
}  // namespace Acts

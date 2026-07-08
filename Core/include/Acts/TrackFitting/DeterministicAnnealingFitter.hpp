// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/ActorList.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/PropagatorOptions.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/DafActor.hpp"
#include "Acts/Utilities/AnnealingUtility.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/HashedString.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"

#include <memory>
#include <optional>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace Acts {

/// @addtogroup track_fitting
/// @{

/// Deterministic Annealing Filter implementation.
///
/// @tparam propagator_t Type of the propagation class, usually an instance of
///         @ref Acts::Propagator
///
/// Runs the Kalman filter/smoother iteratively over a cooling temperature
/// schedule. At each surface with competing measurement candidates, every
/// candidate is assigned a soft chi2/temperature-based weight
/// (@ref AnnealingUtility), and the candidates are combined into a single
/// effective measurement before the usual gain-matrix update runs -- so the
/// update and smoother are reused from @ref KalmanFitter unmodified. Weights
/// sharpen toward a hard assignment as the temperature drops.
///
/// The caller supplies the competing candidate source links pre-grouped by
/// surface; unlike @ref CombinatorialKalmanFilter, the full candidate set for
/// every surface must be known up front, since it is reused across the whole
/// annealing loop rather than discovered live during navigation.
template <typename propagator_t, typename traj_t>
class DeterministicAnnealingFitter {
  /// The navigator type
  using DafNavigator = typename propagator_t::Navigator;

  /// The navigator has DirectNavigator type or not
  static constexpr bool isDirectNavigator =
      std::is_same_v<DafNavigator, DirectNavigator>;

  using Actor = detail::DafActor<traj_t>;

 public:
  /// Constructor with propagator and logger
  /// @param pPropagator Propagator instance for track propagation
  /// @param _logger Logger for diagnostic output
  explicit DeterministicAnnealingFitter(
      propagator_t pPropagator,
      std::unique_ptr<const Logger> _logger = getDefaultLogger(
          "DeterministicAnnealingFitter", Logging::INFO))
      : m_propagator(std::move(pPropagator)),
        m_logger{std::move(_logger)},
        m_actorLogger{m_logger->cloneWithSuffix("Actor")} {}

 private:
  /// The propagator for the transport and material update
  propagator_t m_propagator;

  /// The logger instance
  std::unique_ptr<const Logger> m_logger;
  std::unique_ptr<const Logger> m_actorLogger;

  const Logger& logger() const { return *m_logger; }

 public:
  /// Fit implementation running the annealing loop over the flat-navigator
  /// propagator.
  ///
  /// @tparam track_container_t Type of the track container
  ///
  /// @param measurementGroups The competing candidate source links, grouped
  /// by the surface they are found on. A surface with a single candidate is
  /// valid and simply reduces to a Kalman-equivalent update.
  /// @param sParameters The initial track parameters
  /// @param dafOptions Options steering the fit
  /// @param trackContainer Input track container storage to append into
  ///
  /// @return the output as an output track
  template <TrackContainerFrontend track_container_t>
  Result<typename track_container_t::TrackProxy> fit(
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const BoundTrackParameters& sParameters,
      const DafFitterOptions<traj_t>& dafOptions,
      track_container_t& trackContainer) const {
    return fit_impl(measurementGroups, sParameters, dafOptions, nullptr,
                    trackContainer);
  }

  /// Fit implementation running the annealing loop over the DirectNavigator
  /// propagator.
  ///
  /// @tparam track_container_t Type of the track container
  ///
  /// @param measurementGroups The competing candidate source links, grouped
  /// by the surface they are found on.
  /// @param sParameters The initial track parameters
  /// @param dafOptions Options steering the fit
  /// @param sSequence surface sequence used to initialize a DirectNavigator
  /// @param trackContainer Input track container storage to append into
  ///
  /// @return the output as an output track
  template <TrackContainerFrontend track_container_t>
  Result<typename track_container_t::TrackProxy> fit(
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const BoundTrackParameters& sParameters,
      const DafFitterOptions<traj_t>& dafOptions,
      const std::vector<const Surface*>& sSequence,
      track_container_t& trackContainer) const
    requires(isDirectNavigator)
  {
    return fit_impl(measurementGroups, sParameters, dafOptions, &sSequence,
                    trackContainer);
  }

 private:
  auto make_propagator_options(
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const DafFitterOptions<traj_t>& dafOptions,
      const std::vector<const Surface*>* sSequence,
      DafCandidateMap& candidateState, traj_t& scratchStates) const {
    using DafActorT = Actor;
    using Actors = ActorList<DafActorT>;
    using PropagatorOptions = typename propagator_t::template Options<Actors>;

    PropagatorOptions propagatorOptions(dafOptions.geoContext,
                                        dafOptions.magFieldContext);
    propagatorOptions.setPlainOptions(dafOptions.propagatorPlainOptions);

    if constexpr (!isDirectNavigator) {
      for (const auto& [surface, group] : measurementGroups) {
        (void)group;
        propagatorOptions.navigation.appendExternalSurface(*surface);
      }
    } else {
      assert(sSequence != nullptr &&
             "DirectNavigator requires a surface sequence for "
             "DeterministicAnnealingFitter");
      propagatorOptions.navigation.externalSurfaces = *sSequence;
    }

    auto& dafActor = propagatorOptions.actorList.template get<DafActorT>();
    dafActor.inputMeasurementGroups = measurementGroups;
    dafActor.multipleScattering = dafOptions.multipleScattering;
    dafActor.energyLoss = dafOptions.energyLoss;
    dafActor.freeToBoundCorrection = dafOptions.freeToBoundCorrection;
    dafActor.calibrationContext = &dafOptions.calibrationContext.get();
    dafActor.extensions = dafOptions.extensions;
    dafActor.actorLogger = m_actorLogger.get();
    dafActor.candidateState = &candidateState;
    dafActor.scratchStates = &scratchStates;

    return propagatorOptions;
  }

  template <typename propagator_options_t,
            TrackContainerFrontend track_container_t>
  auto filter_impl(const BoundTrackParameters& sParameters,
                   const propagator_options_t& propagatorOptions,
                   track_container_t& trackContainer) const
      -> Result<typename track_container_t::TrackProxy> {
    auto propagatorState = m_propagator.makeState(propagatorOptions);

    auto propagatorInitResult =
        m_propagator.initialize(propagatorState, sParameters);
    if (!propagatorInitResult.ok()) {
      ACTS_DEBUG("Propagation initialization failed: "
                 << propagatorInitResult.error());
      return propagatorInitResult.error();
    }

    auto& dafResult = propagatorState.template get<DafFitterResult<traj_t>>();
    dafResult.fittedStates = &trackContainer.trackStateContainer();

    auto result = m_propagator.propagate(propagatorState);
    if (!result.ok()) {
      ACTS_DEBUG("Propagation failed: " << result.error());
      return result.error();
    }

    if (!dafResult.measurementStates) {
      ACTS_DEBUG("DeterministicAnnealingFitter failed: No measurement "
                 "states found");
      return DafFitterError::NoMeasurementFound;
    }

    auto track = trackContainer.makeTrack();
    track.tipIndex() = dafResult.lastMeasurementIndex;
    if (dafResult.fittedParameters) {
      const auto& params = dafResult.fittedParameters.value();
      track.parameters() = params.parameters();
      track.covariance() = params.covariance().value();
      track.setReferenceSurface(params.referenceSurface().getSharedPtr());
    }

    calculateTrackQuantities(track);

    return track;
  }

  /// Recomputes each candidate group's annealed weights from the just-
  /// smoothed trajectory of @p track, storing them into @p candidateState as
  /// the prior for the *next* forward pass.
  ///
  /// This deliberately uses the smoothed state, not the predicted state seen
  /// during the forward pass: the smoothed state pulls in information from
  /// every hit on the track (including those after a given surface), so it
  /// can meaningfully discriminate candidates even at the first measurement,
  /// where the forward-only predicted state is still dominated by the
  /// (typically loose) seed covariance. The seed itself is never changed
  /// between iterations -- only these weights evolve.
  template <typename track_proxy_t>
  void updateCandidateWeights(
      track_proxy_t track,
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const DafFitterOptions<traj_t>& dafOptions,
      DafCandidateMap& candidateState, const AnnealingUtility& annealing,
      AnnealingUtility::State& annealingState, traj_t& scratchStates) const {
    scratchStates.clear();

    for (const auto ts : track.trackStatesReversed()) {
      if (!ts.hasCalibrated()) {
        continue;
      }
      const Surface* surface = &ts.referenceSurface();
      auto groupIt = measurementGroups.find(surface);
      if (groupIt == measurementGroups.end()) {
        continue;
      }
      const DafMeasurementGroup& group = groupIt->second;

      std::vector<typename traj_t::TrackStateProxy> candidates;
      candidates.reserve(group.size());
      for (std::size_t i = 0; i < group.size(); ++i) {
        TrackStatePropMask mask =
            (i == 0) ? (TrackStatePropMask::Predicted |
                       TrackStatePropMask::Calibrated)
                    : TrackStatePropMask::Calibrated;
        auto candidate = scratchStates.makeTrackState(mask, kTrackIndexInvalid);
        if (i == 0) {
          candidate.predicted() = ts.smoothed();
          candidate.predictedCovariance() = ts.smoothedCovariance();
        } else {
          candidate.shareFrom(candidates.front(), TrackStatePropMask::Predicted);
        }
        dafOptions.extensions.calibrator(dafOptions.geoContext,
                                         dafOptions.calibrationContext.get(),
                                         group[i], candidate);
        candidates.push_back(candidate);
      }

      std::vector<double> chi2s(group.size());
      for (std::size_t i = 0; i < group.size(); ++i) {
        chi2s[i] = dafOptions.extensions.chiSquareCalculator(
            typename traj_t::ConstTrackStateProxy(candidates[i]));
      }

      std::vector<double> weights(group.size());
      for (std::size_t i = 0; i < group.size(); ++i) {
        weights[i] = annealing.getWeight(annealingState, chi2s[i], chi2s);
      }

      DafCandidateState& candState = candidateState[surface];
      candState.weights = weights;
      candState.chi2 = chi2s;
    }
  }

  /// Common fit implementation: runs the outer annealing loop.
  ///
  /// All iterations except the final one run against an internal scratch
  /// track container, so intermediate (discarded) passes never bloat the
  /// caller-supplied output. The final iteration -- the lowest configured
  /// temperature's last inner iteration -- runs directly into the caller's
  /// @p trackContainer and is the one returned.
  template <TrackContainerFrontend track_container_t>
  auto fit_impl(
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const BoundTrackParameters& sParameters,
      const DafFitterOptions<traj_t>& dafOptions,
      const std::vector<const Surface*>* sSequence,
      track_container_t& trackContainer) const
      -> Result<typename track_container_t::TrackProxy> {
    using TrackProxy = typename track_container_t::TrackProxy;

    if (measurementGroups.empty()) {
      return DafFitterError::NoMeasurementFound;
    }

    const std::size_t numTemperatures =
        dafOptions.annealingConfig.setOfTemperatures.size();
    const std::size_t totalIterations =
        numTemperatures * dafOptions.iterationsPerTemperature;
    if (totalIterations == 0 || totalIterations > dafOptions.maxIterations) {
      return DafFitterError::AnnealingDidNotConverge;
    }

    DafCandidateMap candidateState;
    for (const auto& [surface, group] : measurementGroups) {
      DafCandidateState& candState = candidateState[surface];
      candState.candidates = group;
      candState.weights.assign(group.size(),
                               1. / static_cast<double>(group.size()));
      candState.chi2.assign(group.size(), 0.);
    }

    AnnealingUtility annealing(dafOptions.annealingConfig);
    AnnealingUtility::State annealingState;

    traj_t scratchCandidateStates;

    Acts::TrackContainer scratchTrackContainer{VectorTrackContainer{},
                                               traj_t{}};

    // Re-seed each pass from the previous pass's own smoothed trajectory,
    // starting from the caller-supplied seed on the very first pass --
    // mirroring GenFit's DAF (KalmanFitterRefTrack with refitAll: each
    // annealing iteration re-linearizes around its own previous result
    // rather than the original seed). Only having to correct a
    // trajectory that already incorporates the previous pass's weights,
    // instead of re-propagating from a fixed, possibly-rough seed every
    // single pass, limits how far a single pass's still-uncertain weights
    // (e.g. a 50/50 blend on a freshly-encountered ambiguous surface) can
    // drag the fit before the next pass's sharper weights get a chance to
    // correct it. Only the per-candidate weights in candidateState are
    // otherwise carried between passes (see updateCandidateWeights).
    BoundTrackParameters currentSeed = sParameters;

    // Snapshot of the (seed, per-surface weights) that last produced a
    // successful forward+smooth pass. Used to salvage a usable result when a
    // later pass's propagation fails outright -- e.g. a low-momentum curler
    // whose re-seeded pass hits the step limit -- instead of discarding the
    // whole track. This mirrors GenFit's DAF, which returns its last usable
    // estimate (flagged not-converged) rather than failing the fit outright.
    // The salvaged inputs are known-good: they already yielded a successful
    // pass, and both propagation and smoothing are deterministic, so replaying
    // them into the output container reproduces that success.
    std::optional<BoundTrackParameters> lastGoodSeed;
    DafCandidateMap lastGoodCandidateState;

    // Runs one forward+smooth pass into the caller's output container using
    // the given seed and whatever weights candidateState currently holds, then
    // extrapolates to the reference surface. This is the "official" result-
    // producing pass, shared by the normal final iteration and by salvaging.
    auto finalizeInto =
        [&](const BoundTrackParameters& seed) -> Result<TrackProxy> {
      scratchCandidateStates.clear();
      auto propagatorOptions = make_propagator_options(
          measurementGroups, dafOptions, sSequence, candidateState,
          scratchCandidateStates);

      auto forwardResult = filter_impl(seed, propagatorOptions, trackContainer);
      if (!forwardResult.ok()) {
        return forwardResult.error();
      }
      TrackProxy track = forwardResult.value();

      auto smoothRes = dafOptions.extensions.smoother(
          dafOptions.geoContext, trackContainer.trackStateContainer(),
          track.tipIndex(), logger());
      if (!smoothRes.ok()) {
        return smoothRes.error();
      }

      if (trackContainer.hasColumn(hashString("smoothed"))) {
        track.template component<bool, hashString("smoothed")>() = true;
      }

      if (!track.hasReferenceSurface() &&
          dafOptions.referenceSurface != nullptr) {
        typename propagator_t::template Options<> extrapolationOptions(
            dafOptions.geoContext, dafOptions.magFieldContext);
        auto extrapolationResult = extrapolateTrackToReferenceSurface(
            track, *dafOptions.referenceSurface, m_propagator,
            extrapolationOptions, dafOptions.referenceSurfaceStrategy,
            logger());
        if (!extrapolationResult.ok()) {
          return extrapolationResult.error();
        }
      }

      return track;
    };

    // Attempts to recover a result from the last known-good pass after some
    // pass failed. Falls back to returning the failing pass's own error only
    // when there is nothing to salvage (e.g. the very first pass failed).
    auto salvage = [&](std::error_code passError) -> Result<TrackProxy> {
      if (!lastGoodSeed) {
        return passError;
      }
      ACTS_DEBUG("DAF pass failed ("
                 << passError.message()
                 << "); salvaging last successful pass instead of failing the "
                    "fit");
      candidateState = lastGoodCandidateState;
      auto salvaged = finalizeInto(*lastGoodSeed);
      if (!salvaged.ok()) {
        ACTS_DEBUG("Salvage pass unexpectedly failed: " << salvaged.error());
        return salvaged.error();
      }
      return salvaged;
    };

    for (std::size_t iter = 0; iter < totalIterations; ++iter) {
      const bool isFinalIteration = (iter + 1 == totalIterations);

      if (isFinalIteration) {
        auto finalTrack = finalizeInto(currentSeed);
        if (finalTrack.ok()) {
          return finalTrack.value();
        }
        ACTS_DEBUG("Final DAF pass failed: " << finalTrack.error());
        return salvage(finalTrack.error());
      }

      scratchCandidateStates.clear();
      auto propagatorOptions = make_propagator_options(
          measurementGroups, dafOptions, sSequence, candidateState,
          scratchCandidateStates);

      scratchTrackContainer.clear();
      auto forwardResult = filter_impl(currentSeed, propagatorOptions,
                                       scratchTrackContainer);
      if (!forwardResult.ok()) {
        ACTS_DEBUG("DAF pass " << iter << " failed: " << forwardResult.error());
        return salvage(forwardResult.error());
      }
      auto forwardTrack = forwardResult.value();

      auto smoothRes = dafOptions.extensions.smoother(
          dafOptions.geoContext, scratchTrackContainer.trackStateContainer(),
          forwardTrack.tipIndex(), logger());
      if (!smoothRes.ok()) {
        ACTS_DEBUG("Smoothing step failed on pass " << iter << ": "
                                                    << smoothRes.error());
        return salvage(smoothRes.error());
      }

      // This pass succeeded: remember its (seed, weights) as the fallback for
      // any later pass that fails. Captured now, before both are updated below
      // for the next pass, so they reflect exactly what this pass consumed.
      lastGoodSeed = currentSeed;
      lastGoodCandidateState = candidateState;

      // Carry the smoothed trajectory's first-measurement state forward as
      // next pass's seed. If none can be found (shouldn't happen: filter_impl
      // already requires at least one measurement state), keep the previous
      // seed rather than fail the whole fit over a re-seeding nicety.
      auto firstMeasurement = findFirstMeasurementState(forwardTrack);
      if (firstMeasurement.ok()) {
        const auto& fm = *firstMeasurement;
        currentSeed = BoundTrackParameters(
            fm.referenceSurface().getSharedPtr(), fm.smoothed(),
            fm.smoothedCovariance(), currentSeed.particleHypothesis());
      } else {
        ACTS_DEBUG("Could not find first measurement state to re-seed next "
                   "pass, reusing previous seed: "
                   << firstMeasurement.error());
      }

      updateCandidateWeights(forwardTrack, measurementGroups, dafOptions,
                             candidateState, annealing, annealingState,
                             scratchCandidateStates);

      if ((iter + 1) % dafOptions.iterationsPerTemperature == 0) {
        annealing.anneal(annealingState);
      }
    }

    return DafFitterError::AnnealingDidNotConverge;
  }
};

/// @}

}  // namespace Acts

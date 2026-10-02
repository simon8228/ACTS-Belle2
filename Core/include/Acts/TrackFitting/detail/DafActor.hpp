// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/MultiTrajectoryHelpers.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/detail/CorrectedTransformationFreeToBound.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/Surfaces/LineBounds.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/DafReferenceState.hpp"
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

  /// Previous reference state's parameters and validity flag. Only used while
  /// DafActor::recordReference is set: a segment's delta state is a difference
  /// against the previous surface, so the recorder has to remember it. The
  /// process noise needs no such memory -- the stepping covariance is reset to
  /// zero at every recorded surface, so it accumulates exactly one segment's
  /// noise. See DafReferenceState.
  BoundVector lastReferenceParameters = BoundVector::Zero();
  bool hasLastReference = false;

  /// Material crossed since the last recorded reference surface, accumulated
  /// EXACTLY as the process noise is: reset wherever state.stepping.cov is
  /// reset, so it pairs with the Q of the same segment. Pairing material
  /// recorded AT a surface with that surface's own Q would compare mismatched
  /// things -- a surface's post-update material lands in the NEXT segment's Q,
  /// which is what makes "hasMaterial but Q == 0" look like a bug when it is
  /// not.
  double segmentThicknessInX0 = 0.;
  double segmentELoss = 0.;
  double segmentVariancePhi = 0.;
  double segmentVarianceTheta = 0.;
  double segmentVarianceQoverP = 0.;
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

  /// Allows retrieving the competing candidate source links for a surface.
  ///
  /// NON-OWNING, and deliberately so. Propagator::makeState copies the whole
  /// actor list into the propagator state, so a by-value map here is deep-
  /// copied -- every group, every SourceLink -- on EVERY propagation. That is
  /// tolerable at one propagation per pass and is not at ~110, which is what
  /// the piecewise reference recording does. The map lives in fit_impl for the
  /// whole fit, so it comfortably outlives any propagation that reads it.
  const std::unordered_map<const Surface*, DafMeasurementGroup>*
      inputMeasurementGroups = nullptr;

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

  /// Whether this is the first annealing pass, i.e. the one whose candidate
  /// weights are the driver's uniform 1/N rather than something derived from a
  /// previous pass's smoothed trajectory. Only then are the drift-circle
  /// initial weights below applied.
  bool firstPass = false;

  /// Record a reference trajectory instead of filtering.
  ///
  /// With this set the actor still propagates and still calibrates every
  /// candidate on every measurement surface, but it performs no weighted
  /// combination, no Kalman update, and -- crucially -- never pushes a
  /// filtered state back into the stepper. The propagation therefore stays on
  /// the trajectory it started from, which is exactly GenFit's
  /// KalmanFitterRefTrack::prepareTrack: the pass produces the linearisation
  /// (reference parameters, F, N, c) and the reference-calibrated candidates
  /// that detail::dafReferenceFilter then filters without propagating.
  bool recordReference = false;

  /// Where the recorded reference states are appended, in propagation order.
  /// Must be non-null when recordReference is set.
  DafReferenceTrajectory* referenceTrajectory{nullptr};

  /// Enables GenFit's drift-circle initial weighting on straw surfaces, as a
  /// multiple of the straw's radial CELL SPACING (0 = off, keep the uniform
  /// 1/N).
  ///
  /// A 50/50 blend of two mirror hypotheses puts the effective measurement on
  /// the wire with an uninflated covariance, so the first pass is pulled toward
  /// the wire by an amount that grows with the drift radius. GenFit counteracts
  /// this by starting large-radius hits at a low weight
  /// (cdc/dataobjects/src/CDCRecoHit.cc):
  ///
  ///     rMax   = 1.5 * cell spacing      (15 mm inner superlayer, 27 mm rest)
  ///     weight = 0.5 * max(0, 1 - rbar/rMax)^2      (per candidate)
  ///
  /// so the weights sum to <= 1 and the combiner's R_eff = R / sum(w) is
  /// inflated exactly where the blend is least trustworthy.
  ///
  /// The cell spacing is recovered from the straw's own bounds rather than
  /// configured separately: the Belle2 geometry builds every wire's LineBounds
  /// as cell/2 - kStrawFieldWireClearance (Belle2.cpp), so
  ///
  ///     cell spacing = 2 * (boundsR + kStrawFieldWireClearance)
  ///     rMax         = factor * (boundsR + kStrawFieldWireClearance)
  ///
  /// and a factor of 3 reproduces GenFit's 1.5 * cell spacing exactly, for
  /// both the 10 mm inner superlayer and the 18 mm cells outside it, with no
  /// per-layer branching. The factor stays a knob so the margin can be tuned.
  double strawInitialWeightRadiusFactor = 0.;

  /// Radial clearance between a drift cell's boundary and the straw surface's
  /// LineBounds radius, i.e. boundsR = cell/2 - this. Set by the Belle2 CDC
  /// geometry builder (Examples/Detectors/Belle2/src/Belle2.cpp) to keep
  /// neighbouring wire surfaces from overlapping; reused here only to undo that
  /// offset when recovering the cell spacing above.
  static constexpr double kStrawFieldWireClearance = 0.8 * UnitConstants::mm;

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

    // ACTS_DN_HELIX_RANK: hand the direct navigator the local bending,
    // q/p * B, for ranking its look-ahead candidates along the helix.
    if constexpr (requires { state.navigation.qopField; }) {
      if (detail::directNavigatorHelixRank()) {
        auto field =
            stepper.getField(state.stepping, stepper.position(state.stepping));
        if (field.ok()) {
          state.navigation.qopField =
              stepper.qOverP(state.stepping) * (*field);
          state.navigation.hasQopField = true;
        }
      }
    }

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
          << (inputMeasurementGroups != nullptr
                  ? inputMeasurementGroups->size()
                  : 0u)
          << " measurementStates=" << result.measurementStates
          << " currentSurface="
          << (surface != nullptr ? surface->geometryId().value() : 0)
          << " targetSurface="
          << (targetSurface != nullptr ? targetSurface->geometryId().value()
                                        : 0));
      return DafFitterError::PropagationStalled;
    }

    const bool isTrackComplete =
        inputMeasurementGroups != nullptr &&
        result.measurementStates == inputMeasurementGroups->size();
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

    if (inputMeasurementGroups == nullptr) {
      return Result<void>::success();
    }
    const auto groupIt = inputMeasurementGroups->find(&surface);
    if (groupIt != inputMeasurementGroups->end()) {
      const DafMeasurementGroup& group = groupIt->second;
      ACTS_VERBOSE("Measurement surface " << surface.geometryId()
                                          << " detected with "
                                          << group.size() << " candidate(s).");

      stepper.transportCovarianceToBound(state.stepping, surface,
                                         freeToBoundCorrection);

      accumulateMaterial(
          Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::PreUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger()),
          surface, stepper.position(state.stepping), result);

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
      // While recording a reference trajectory the calibrated candidates have
      // to survive the whole propagation: the filter phase consumes them
      // afterwards, indexed by DafReferenceState::candidateIndices. In the
      // normal filtering mode they are consumed on the spot and the store is
      // recycled per surface.
      if (!recordReference) {
        scratchStates->clear();
      }

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
        // A calibrator may need to know which surface it is calibrating on --
        // a drift-chamber one has to, since the x-t relation depends on the
        // incidence angle against the wire. The scratch states are not part of
        // the trajectory, so nothing else sets this for them, and without it
        // referenceSurface() dereferences an unset pointer.
        candidate.setReferenceSurface(surface.getSharedPtr());
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

      // Drift-circle initial weights (see strawInitialWeightRadiusFactor).
      // Only on the first pass: from the second onward the weights carry real
      // information from the previous pass's smoothed trajectory, which must
      // not be overwritten by a geometric prior.
      if (firstPass && strawInitialWeightRadiusFactor > 0. &&
          candidates.size() > 1 && surface.type() == Surface::Straw) {
        const auto* lineBounds =
            dynamic_cast<const LineBounds*>(&surface.bounds());
        if (lineBounds != nullptr) {
          // Undo the geometry's field-wire clearance to recover half the radial
          // cell spacing, which is what GenFit's rMax is a multiple of.
          const double rMax =
              strawInitialWeightRadiusFactor *
              (lineBounds->get(LineBounds::eR) + kStrawFieldWireClearance);
          if (rMax > 0.) {
            // Mean drift radius of the competing hypotheses, i.e. GenFit's
            // (mL + mR) / 2 -- the calibrated values are the signed local
            // coordinate, so take magnitudes.
            double meanRadius = 0.;
            for (const auto& candidate : candidates) {
              meanRadius += std::abs(candidate.effectiveCalibrated()[0]);
            }
            meanRadius /= static_cast<double>(candidates.size());

            const double reduced = std::max(0., 1. - meanRadius / rMax);
            const double weight = 0.5 * reduced * reduced;
            ACTS_VERBOSE("DafActor: drift-circle initial weight on surface "
                         << surface.geometryId() << ": rbar=" << meanRadius
                         << " rMax=" << rMax << " -> w=" << weight
                         << " per candidate");
            candState.weights.assign(candidates.size(), weight);
          }
        }
      }

      // Reference-recording mode stops here: no weighted combination, no
      // Kalman update, and above all no stepper.update() below -- the
      // propagation has to stay on the reference trajectory instead of
      // following the filtered estimate. That single omission is what turns
      // this actor from GenFit's plain KalmanFitter into its
      // KalmanFitterRefTrack::prepareTrack.
      if (recordReference) {
        recordReferenceState(surface, boundParams.parameters(), jacobian,
                             pathLength, state.stepping.cov, result,
                             /*hasMeasurement=*/true, surfaceHasMaterial,
                             /*isHole=*/false, &candidates);
        // Restart the noise accumulation for the next segment. Safe because the
        // recording pass never uses the covariance for anything -- no update, no
        // gain -- and the propagated path does not depend on it.
        state.stepping.cov.setZero();

        auto refTypeFlags = trackStateProxy.typeFlags();
        refTypeFlags.setHasParameters();
        if (surfaceHasMaterial) {
          refTypeFlags.setHasMaterial();
        }
        result.lastTrackIndex = trackStateProxy.index();
        result.lastMeasurementIndex = result.lastTrackIndex;
        ++result.measurementStates;
        ++result.processedStates;
        result.measurementHoles = result.missedActiveSurfaces.size();

        accumulateMaterial(
            Acts::detail::performMaterialInteraction(
            state, stepper, surface,
            Acts::detail::determineMaterialUpdateMode(
                state, navigator, MaterialUpdateMode::PostUpdate),
            NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
            logger()),
            surface, stepper.position(state.stepping), result);
        return Result<void>::success();
      }

      extensions.weightedCombiner(constCandidates, candState.weights,
                                  trackStateProxy);

      // Surfaces with a real multi-candidate decision log at DEBUG; the
      // (common) single-candidate case repeats near-identically every
      // annealing pass and is only VERBOSE-level detail.
      const bool hasCandidateChoice = candidates.size() > 1;
      if (logger().doPrint(hasCandidateChoice ? Logging::DEBUG
                                              : Logging::VERBOSE)) {
        std::stringstream candSs;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
          candSs << " [w=" << candState.weights[i] << " meas="
                 << candidates[i].effectiveCalibrated().transpose() << "]";
        }
        std::stringstream msg;
        msg << "DafActor: combined " << candidates.size()
            << " candidate(s) on surface " << surface.geometryId()
            << " predicted=" << trackStateProxy.predicted().transpose()
            << " candidates:" << candSs.str() << " -> combined meas="
            << trackStateProxy.effectiveCalibrated().transpose()
            << " combined covariance diag="
            << trackStateProxy.effectiveCalibratedCovariance()
                   .diagonal()
                   .transpose()
            << " measdim=" << trackStateProxy.calibratedSize();
        if (hasCandidateChoice) {
          ACTS_DEBUG(msg.str());
        } else {
          ACTS_VERBOSE(msg.str());
        }
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

      if (logger().doPrint(hasCandidateChoice ? Logging::DEBUG
                                              : Logging::VERBOSE)) {
        const double predictedP =
            1. / std::abs(trackStateProxy.predicted()[eBoundQOverP]);
        const double predictedPt =
            predictedP * std::sin(trackStateProxy.predicted()[eBoundTheta]);
        const double filteredP =
            1. / std::abs(trackStateProxy.filtered()[eBoundQOverP]);
        const double filteredPt =
            filteredP * std::sin(trackStateProxy.filtered()[eBoundTheta]);
        std::stringstream msg;
        msg << "DafActor: Kalman update on surface " << surface.geometryId()
            << " predicted p=" << predictedP << " pT=" << predictedPt
            << " -> filtered p=" << filteredP << " pT=" << filteredPt
            << " (ratio p=" << filteredP / predictedP << ")";
        if (hasCandidateChoice) {
          ACTS_DEBUG(msg.str());
        } else {
          ACTS_VERBOSE(msg.str());
        }
      }

      ACTS_VERBOSE("Filtering step successful, updated parameters are:\n"
                   << trackStateProxy.filtered().transpose());
      stepper.update(state.stepping,
                     MultiTrajectoryHelpers::freeFiltered(
                         state.options.geoContext, trackStateProxy),
                     trackStateProxy.filtered(),
                     trackStateProxy.filteredCovariance(), surface);
      ++result.measurementStates;

      accumulateMaterial(
          Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::PostUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger()),
          surface, stepper.position(state.stepping), result);
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

      if (recordReference) {
        recordReferenceState(surface, boundParams.parameters(), jacobian,
                             pathLength, state.stepping.cov, result,
                             /*hasMeasurement=*/false, surfaceHasMaterial,
                             /*isHole=*/surfaceIsSensitive &&
                                 precedingMeasurementExists,
                             /*candidates=*/nullptr);
        // As above: restart the noise accumulation for the next segment.
        state.stepping.cov.setZero();
      }

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

      accumulateMaterial(
          Acts::detail::performMaterialInteraction(
          state, stepper, surface,
          Acts::detail::determineMaterialUpdateMode(
              state, navigator, MaterialUpdateMode::FullUpdate),
          NoiseUpdateMode::addNoise, multipleScattering, energyLoss,
          logger()),
          surface, stepper.position(state.stepping), result);
    }

    return Result<void>::success();
  }

  /// Append one entry to the reference trajectory, deriving the segment's
  /// process noise and delta state from the previously recorded entry.
  ///
  /// @param surface The surface the reference state sits on
  /// @param parameters Reference parameters on that surface
  /// @param jacobian Transport Jacobian accumulated since the previous entry
  /// @param pathLength Accumulated path length at this surface
  /// @param covAtSurface Stepping covariance at this surface, which IS this
  ///        segment's process noise: it is reset to zero at every recorded
  ///        surface, so it accumulates over exactly one segment.
  /// @param result The actor result carrying the previous entry
  /// @param hasMeasurement Whether the surface carries a measurement group
  /// @param hasMaterial Whether the surface has material
  /// @param isHole Whether the surface is a hole
  /// @param candidates The surface's calibrated candidates, or nullptr
  /// Fold one material interaction into the current segment's running totals.
  ///
  /// x/X0 is taken from the surface's own slab rather than inferred from the
  /// returned variances, because that is exactly what separates "the material
  /// map carries no thickness here" from "a real slab produced no noise" --
  /// a distinction the variances alone cannot make.
  void accumulateMaterial(const Acts::detail::PointwiseMaterialEffects& effects,
                          const Surface& surface, const Vector3& position,
                          result_type& result) const {
    result.segmentELoss += effects.eLoss;
    result.segmentVariancePhi += effects.variancePhi;
    result.segmentVarianceTheta += effects.varianceTheta;
    result.segmentVarianceQoverP += effects.varianceQoverP;
    const auto* material = surface.surfaceMaterial();
    if (material != nullptr) {
      result.segmentThicknessInX0 +=
          material->materialSlab(position).thicknessInX0();
    }
  }

  void recordReferenceState(
      const Surface& surface, const BoundVector& parameters,
      const BoundMatrix& jacobian, double pathLength,
      const BoundSquareMatrix& covAtSurface, result_type& result,
      bool hasMeasurement, bool hasMaterial, bool isHole,
      const std::vector<typename traj_t::TrackStateProxy>* candidates) const {
    assert(referenceTrajectory != nullptr &&
           "DafActor::recordReference requires a reference trajectory");

    DafReferenceState record;
    record.surface = &surface;
    record.parameters = parameters;
    record.pathLength = pathLength;
    record.hasMeasurement = hasMeasurement;
    record.hasMaterial = hasMaterial;
    record.isHole = isHole;

    if (result.hasLastReference) {
      record.jacobian = jacobian;

      // Exactly GenFit's definition, see RKTrackRep::getForwardJacobianAndNoise:
      // the constant that makes the affine model reproduce the reference
      // transport, lastEndState - jacobian * lastStartState.
      record.deltaState = parameters - jacobian * result.lastReferenceParameters;
    } else {
      // First entry: jacobian is the transport from the propagation start, which
      // detail::dafReferenceFilter uses to bring the seed covariance onto this
      // surface rather than pretending a covariance defined at the seed surface
      // is valid here.
      record.jacobian = jacobian;
    }

    // The stepping covariance was zeroed at the previous recorded surface (see
    // the call sites), so what it holds now is precisely this segment's process
    // noise -- no subtraction involved.
    //
    // It used to be recovered as N_k = C_k - F*C_k-1*F^T, on the theory that a
    // zero-covariance start makes C pure noise and the difference therefore
    // well-conditioned. That was wrong: C still ACCUMULATES, so by the outer
    // layers the running total dwarfs the per-segment increment, the difference
    // is a small residue of two large matrices, and on roughly half the
    // segments it came out negative and was clamped to zero -- leaving those
    // segments with no material noise at all.
    record.noise = covAtSurface;

    // Same segment as `noise`, so the two are directly comparable -- then
    // reset, exactly as the caller resets state.stepping.cov straight after.
    record.thicknessInX0 = result.segmentThicknessInX0;
    record.energyLoss = result.segmentELoss;
    record.varianceTheta = result.segmentVarianceTheta;
    record.varianceQoverP = result.segmentVarianceQoverP;
    result.segmentThicknessInX0 = 0.;
    result.segmentELoss = 0.;
    result.segmentVariancePhi = 0.;
    result.segmentVarianceTheta = 0.;
    result.segmentVarianceQoverP = 0.;

    if (candidates != nullptr) {
      record.candidateIndices.reserve(candidates->size());
      for (const auto& candidate : *candidates) {
        record.candidateIndices.push_back(candidate.index());
      }
    }

    // The process noise is the one quantity in this scheme with no ACTS
    // equivalent to check against -- it is reconstructed by differencing rather
    // than read off, so dump it when asked. sqrt(N(theta,theta)) should sit at
    // the multiple-scattering angle of one layer, 13.6 MeV/(beta c p) *
    // sqrt(x/X0): order 1e-4 rad for a 1 GeV track through a CDC layer. A zero
    // or denormal value means the recovery is broken and the filter is running
    // without material noise, which reads exactly as degraded resolution.
    if (logger().doPrint(Logging::DEBUG)) {
      const BoundVector sigma = record.noise.diagonal().cwiseSqrt();
      ACTS_DEBUG("DafActor: reference state on surface "
                 << surface.geometryId() << " s=" << pathLength
                 << " meas=" << hasMeasurement << " mat=" << hasMaterial
                 << " sqrt(diag N): loc0=" << sigma[eBoundLoc0]
                 << " loc1=" << sigma[eBoundLoc1]
                 << " phi=" << sigma[eBoundPhi]
                 << " theta=" << sigma[eBoundTheta]
                 << " qop=" << sigma[eBoundQOverP]);
    }

    referenceTrajectory->push_back(std::move(record));

    result.lastReferenceParameters = parameters;
    result.hasLastReference = true;
  }
};

}  // namespace detail
}  // namespace Acts

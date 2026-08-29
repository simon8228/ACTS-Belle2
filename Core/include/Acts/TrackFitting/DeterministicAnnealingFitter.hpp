// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MeasurementHelpers.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/SubspaceHelpers.hpp"
#include "Acts/EventData/TrackParameterHelpers.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/ActorList.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/PropagatorOptions.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/Chi2SurvivalFunction.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "Acts/TrackFitting/detail/DafActor.hpp"
#include "Acts/TrackFitting/detail/DafReferenceFilter.hpp"
#include "Acts/TrackFitting/detail/DafReferenceState.hpp"
#include "Acts/Utilities/AnnealingUtility.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/HashedString.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  /// GenFit's MeasuredStateOnPlane::blowUpCov, in its own order: reset the
  /// off-diagonals, scale, then clip every variance to a ceiling (its
  /// blowUpMaxVal, 1e6).
  ///
  /// The clip is what makes a large scaling survivable, and why GenFit can use
  /// 1000 where a uniform ACTS scaling cannot. GenFit's plane state
  /// (q/p, u', v', u, v) is entirely unbounded, so a huge variance just means
  /// "no information". An ACTS bound state carries phi and theta, which are
  /// periodic: past sigma ~ pi the prior is not merely flat but meaningless,
  /// the first update sends the direction anywhere, and normalizePhiTheta folds
  /// it back with a phi flip. Ceilings are per parameter so loc0/loc1/qop can
  /// take the full blow-up while the angles stay well inside the branch cut.
  ///
  /// Shared by the inter-pass re-seed and the backward filter's starting state
  /// so the two cannot end up with different wrap protection.
  /// @param ceiling Uniform variance ceiling, 0 to disable
  /// @param ceilings Per-parameter ceilings, overriding @p ceiling where set
  void blowUpCovariance(BoundSquareMatrix& cov, double scaling,
                        bool resetOffDiagonals, double ceiling,
                        const std::array<double, eBoundSize>& ceilings) const {
    if (resetOffDiagonals) {
      const BoundVector diag = cov.diagonal();
      cov.setZero();
      cov.diagonal() = diag;
    }
    if (scaling != 1.) {
      cov *= scaling;
    }
    for (std::size_t i = 0; i < eBoundSize; ++i) {
      const double limit = ceilings[i] > 0. ? ceilings[i] : ceiling;
      if (limit > 0. && cov(i, i) > limit) {
        // Scale the row and column rather than only the diagonal, so a clipped
        // variance cannot leave a correlation coefficient above 1 and the
        // matrix stays positive definite.
        const double f = std::sqrt(limit / cov(i, i));
        cov.row(i) *= f;
        cov.col(i) *= f;
      }
    }
  }

  auto make_propagator_options(
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const DafFitterOptions<traj_t>& dafOptions,
      const std::vector<const Surface*>* sSequence,
      DafCandidateMap& candidateState, traj_t& scratchStates,
      bool firstPass = false, bool recordReference = false,
      DafReferenceTrajectory* referenceTrajectory = nullptr,
      Direction direction = Direction::Forward()) const {
    using DafActorT = Actor;
    using Actors = ActorList<DafActorT>;
    using PropagatorOptions = typename propagator_t::template Options<Actors>;

    PropagatorOptions propagatorOptions(dafOptions.geoContext,
                                        dafOptions.magFieldContext);
    propagatorOptions.setPlainOptions(dafOptions.propagatorPlainOptions);
    // Set after setPlainOptions, which would otherwise overwrite it with the
    // forward default.
    propagatorOptions.direction = direction;

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
    dafActor.inputMeasurementGroups = &measurementGroups;
    dafActor.multipleScattering = dafOptions.multipleScattering;
    dafActor.energyLoss = dafOptions.energyLoss;
    dafActor.freeToBoundCorrection = dafOptions.freeToBoundCorrection;
    dafActor.calibrationContext = &dafOptions.calibrationContext.get();
    dafActor.extensions = dafOptions.extensions;
    dafActor.actorLogger = m_actorLogger.get();
    dafActor.candidateState = &candidateState;
    dafActor.scratchStates = &scratchStates;
    dafActor.firstPass = firstPass;
    dafActor.stallStepLimit = dafOptions.stallStepLimit;
    dafActor.strawInitialWeightRadiusFactor =
        dafOptions.strawInitialWeightRadiusFactor;
    dafActor.recordReference = recordReference;
    dafActor.referenceTrajectory = referenceTrajectory;

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

  /// Runs a reference pass: propagate without filtering, recording the
  /// linearisation (reference parameters, F, N, c) and the reference-calibrated
  /// candidates that detail::dafReferenceFilter then consumes.
  ///
  /// @param sParameters Parameters the reference is propagated from. Must carry
  ///        a ZERO covariance -- see DafActor::recordReferenceState.
  /// @param propagatorOptions Options with DafActor::recordReference set
  /// @param recordingStates Throwaway trajectory for the actor's own track
  ///        states; the reference records are what matter and go elsewhere
  template <typename propagator_options_t>
  Result<void> record_impl(const BoundTrackParameters& sParameters,
                           const propagator_options_t& propagatorOptions,
                           traj_t& recordingStates) const {
    auto propagatorState = m_propagator.makeState(propagatorOptions);

    auto propagatorInitResult =
        m_propagator.initialize(propagatorState, sParameters);
    if (!propagatorInitResult.ok()) {
      ACTS_DEBUG("Reference propagation initialization failed: "
                 << propagatorInitResult.error());
      return propagatorInitResult.error();
    }

    auto& dafResult = propagatorState.template get<DafFitterResult<traj_t>>();
    dafResult.fittedStates = &recordingStates;

    auto result = m_propagator.propagate(propagatorState);
    if (!result.ok()) {
      ACTS_DEBUG("Reference propagation failed: " << result.error());
      return result.error();
    }

    if (!dafResult.measurementStates) {
      ACTS_DEBUG("Reference pass found no measurement states");
      return DafFitterError::NoMeasurementFound;
    }

    return Result<void>::success();
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
  ///
  /// @p reference is the trajectory the just-finished pass linearised about, or
  /// null when the pass ran the running-state filter. When it is supplied and
  /// DafFitterOptions::referenceWeightCalibration is set, the candidates are not
  /// re-calibrated here at all: the ones the reference pass already calibrated
  /// are reused, with only their expansion point swapped for the smoothed state
  /// before scoring. That is GenFit's arrangement -- one calibration per pass,
  /// shared by the filter and the weights -- and it keeps the drift length the
  /// weight chi2 is computed from equal to the one the filter actually fitted.
  /// See DafFitterOptions::referenceWeightCalibration.
  /// @return what this pass observed, for the convergence tests -- the largest
  /// weight change (GenFit's maxAbsChange) and the smoothed weighted chi2/ndf.
  template <typename track_proxy_t>
  DafPassSummary updateCandidateWeights(
      track_proxy_t track,
      const std::unordered_map<const Surface*, DafMeasurementGroup>&
          measurementGroups,
      const DafFitterOptions<traj_t>& dafOptions,
      DafCandidateMap& candidateState, const AnnealingUtility& annealing,
      AnnealingUtility::State& annealingState, traj_t& scratchStates,
      const DafReferenceTrajectory* reference = nullptr) const {
    DafPassSummary summary;

    // Index the reference by surface, so the loop below can ask for a surface's
    // recorded candidates in one lookup rather than scanning the trajectory.
    std::unordered_map<const Surface*, const std::vector<std::size_t>*>
        recordedCandidates;
    const bool reuseRecorded = reference != nullptr &&
                               dafOptions.referenceLinearization &&
                               dafOptions.referenceWeightCalibration;
    if (reuseRecorded) {
      for (const DafReferenceState& ref : *reference) {
        if (ref.hasMeasurement && !ref.candidateIndices.empty()) {
          recordedCandidates.emplace(ref.surface, &ref.candidateIndices);
        }
      }
    } else {
      // Only safe when nothing here points into the store: reusing the recorded
      // candidates means reusing the very rows this would drop.
      scratchStates.clear();
    }

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

      // The state the candidates are scored against. By default the smoothed
      // state; optionally that state with this surface's own measurement
      // subtracted again, so a surface does not vote on itself. See
      // DafFitterOptions::useUnbiasedSmoothedState.
      BoundVector refParams = ts.smoothed();
      BoundSquareMatrix refCov = ts.smoothedCovariance();
      if (dafOptions.useUnbiasedSmoothedState) {
        visit_measurement(ts.calibratedSize(), [&](auto N) {
          constexpr std::size_t kMeasurementSize = decltype(N)::value;
          const auto meas = ts.template calibrated<kMeasurementSize>();
          const auto measCov =
              ts.template calibratedCovariance<kMeasurementSize>();
          const FixedBoundSubspaceHelper<kMeasurementSize> subspaceHelper(
              ts.template projectorSubspaceIndices<kMeasurementSize>());
          const auto H = subspaceHelper.projector();

          const SquareMatrix<kMeasurementSize> measInfo = measCov.inverse();
          const BoundSquareMatrix infoSmoothed = refCov.inverse();
          const BoundSquareMatrix infoUnbiased =
              infoSmoothed - H.transpose() * measInfo * H;

          // Catastrophic cancellation whenever the measurement dominates the
          // prior: the subtraction then leaves a matrix that is not positive
          // definite, and the Cholesky decomposition is the cheap way to say
          // so. Keeping the biased state is the safe fallback -- it is what
          // the fitter did before this option existed.
          const Eigen::LLT<BoundSquareMatrix> llt(infoUnbiased);
          if (llt.info() != Eigen::Success) {
            ACTS_VERBOSE("Unbiased smoothed state on surface "
                         << surface->geometryId()
                         << " is not positive definite; scoring against the "
                            "biased smoothed state instead");
            return;
          }
          const BoundSquareMatrix covUnbiased = infoUnbiased.inverse();
          refParams = normalizeBoundParameters(
              covUnbiased * (infoSmoothed * refParams -
                             H.transpose() * measInfo * meas));
          refCov = covUnbiased;
        });
      }

      std::vector<typename traj_t::TrackStateProxy> candidates;
      candidates.reserve(group.size());

      // Reuse this surface's reference-calibrated candidates when the pass
      // recorded them and their count still matches the group -- a mismatch
      // would mean the two are not describing the same surface any more, and
      // silently scoring N weights against M measurements is worse than paying
      // for a re-calibration.
      auto recordedIt = recordedCandidates.find(surface);
      const bool reuseHere = recordedIt != recordedCandidates.end() &&
                             recordedIt->second->size() == group.size();
      if (reuseHere) {
        for (std::size_t index : *recordedIt->second) {
          candidates.push_back(scratchStates.getTrackState(index));
        }
        // The recorded candidates were calibrated on the reference and their
        // predicted state IS the reference state. Move that expansion point to
        // the smoothed state so the chi2 is a residual against the fit, leaving
        // the calibrated measurement itself untouched -- which is the whole
        // point. Candidates past the first share the first one's predicted
        // storage (DafActor sets them up that way), so one write covers all.
        candidates.front().predicted() = refParams;
        candidates.front().predictedCovariance() = refCov;
      } else {
        if (reuseRecorded && recordedIt != recordedCandidates.end()) {
          ACTS_DEBUG("Recorded candidate count on surface "
                     << surface->geometryId() << " ("
                     << recordedIt->second->size() << ") does not match the "
                     << group.size()
                     << " in its measurement group; re-calibrating instead");
        }
        for (std::size_t i = 0; i < group.size(); ++i) {
          TrackStatePropMask mask =
              (i == 0) ? (TrackStatePropMask::Predicted |
                         TrackStatePropMask::Calibrated)
                      : TrackStatePropMask::Calibrated;
          auto candidate =
              scratchStates.makeTrackState(mask, kTrackIndexInvalid);
          // Same reason as in DafActor: a calibrator may need the surface it is
          // calibrating against, and these scratch states are not part of any
          // trajectory that would otherwise set it.
          candidate.setReferenceSurface(surface->getSharedPtr());
          if (i == 0) {
            candidate.predicted() = refParams;
            candidate.predictedCovariance() = refCov;
          } else {
            candidate.shareFrom(candidates.front(),
                                TrackStatePropMask::Predicted);
          }
          dafOptions.extensions.calibrator(dafOptions.geoContext,
                                           dafOptions.calibrationContext.get(),
                                           group[i], candidate);
          candidates.push_back(candidate);
        }
      }

      std::vector<double> chi2s(group.size());
      for (std::size_t i = 0; i < group.size(); ++i) {
        chi2s[i] = dafOptions.extensions.chiSquareCalculator(
            typename traj_t::ConstTrackStateProxy(candidates[i]));
      }

      // GenFit scales every term of the weight, candidates and cut-off alike,
      // by that candidate's own 1/sqrt((2*pi)^d det S). Only collected when
      // asked for: it is a no-op wherever the group shares one S, and paying
      // for a determinant per candidate per pass is not.
      std::vector<double> logNorms;
      if (dafOptions.useCandidateNormalization) {
        logNorms.resize(group.size());
        for (std::size_t i = 0; i < group.size(); ++i) {
          logNorms[i] = dafOptions.extensions.logNormalization(
              typename traj_t::ConstTrackStateProxy(candidates[i]));
        }
      }

      const std::vector<double> weights = annealing.getWeights(
          annealingState, chi2s, logNorms.empty() ? nullptr : &logNorms);

      DafCandidateState& candState = candidateState[surface];
      // Single-candidate surfaces are skipped: their weight is trivially ~1
      // and would only drown out the surfaces where the DAF actually has a
      // choice to make.
      if (group.size() > 1 && logger().doPrint(Logging::DEBUG)) {
        std::stringstream ss;
        for (std::size_t i = 0; i < group.size(); ++i) {
          ss << " [chi2=" << chi2s[i] << " w: " << candState.weights[i]
             << " -> " << weights[i] << "]";
        }
        ACTS_DEBUG("DAF weight update on surface " << surface->geometryId()
                                                   << ":" << ss.str());
      }
      // Read before the new weights overwrite the old ones.
      for (std::size_t i = 0; i < weights.size(); ++i) {
        summary.maxAbsWeightChange =
            std::max(summary.maxAbsWeightChange,
                     std::abs(weights[i] - candState.weights[i]));
        summary.weightedChi2 += weights[i] * chi2s[i];
      }
      summary.ndf += static_cast<double>(ts.calibratedSize());

      candState.weights = weights;
      candState.chi2 = chi2s;
      // Where this pass's own smoothed track crossed the surface, in the same
      // local coordinate the candidates use. Recorded alongside the weights
      // so a trace can show the track moving as the weights that pull it
      // change, rather than only the final trajectory.
      candState.trackLoc0 = ts.smoothed()[eBoundLoc0];

      // The linearisation point this pass used on this surface. Only while a
      // trace is being recorded: it is a per-pass comparison by construction,
      // so a fit that keeps no history has nothing to compare against, and
      // the frame construction below is not free.
      //
      // The PREDICTED state is the one the forward filter expanded F and H
      // about, so it -- not the smoothed state -- is what "the expansion
      // point moved" refers to. The smoothed state is recorded alongside it
      // because the final pass's smoothed trajectory is the closest thing
      // this fit has to the converged track a reference would be built from.
      if (dafOptions.candidateTrace != nullptr) {
        BoundVector linParams = ts.smoothed();
        if (ts.hasPredicted()) {
          linParams = ts.predicted();
        }
        candState.predictedParameters = linParams;
        candState.smoothedParameters = ts.smoothed();
        if (ts.hasJacobian()) {
          candState.jacobian = ts.jacobian();
        }
        const Vector3 direction = makeDirectionFromPhiTheta(
            linParams[eBoundPhi], linParams[eBoundTheta]);
        const Vector3 position = surface->localToGlobal(
            dafOptions.geoContext,
            Vector2(linParams[eBoundLoc0], linParams[eBoundLoc1]), direction);
        candState.globalPosition = position;
        // A straw's measurement frame is built from the track direction, not
        // from the wire alone, so this axis rotates as the fit updates phi
        // and theta -- see DafCandidateState::measurementAxis.
        // Held in a named matrix rather than assigned straight off the
        // by-value return: a .col() of a temporary is legal here but reads
        // like a dangling reference, and this costs nothing.
        const RotationMatrix3 frame = surface->referenceFrame(
            dafOptions.geoContext, position, direction);
        candState.measurementAxis = frame.col(0);
        candState.hasLinearization = true;
      }
    }

    // The five track parameters the measurements had to determine.
    summary.ndf -= 5.;
    return summary;
  }

  /// Report where two recordings of the same trajectory disagree.
  ///
  /// Used by referencePiecewiseSelfCheck: with anchoring off, a piecewise
  /// recording must reproduce the continuous one, because the segment
  /// boundaries fall exactly where the continuous recording already divides the
  /// noise. A difference means the split lost something -- most likely a
  /// surface's post-update material, which a propagation starting on that
  /// surface can skip.
  static void reportReferenceDifference(
      const DafReferenceTrajectory& piecewise,
      const DafReferenceTrajectory& continuous, const Logger& logger) {
    ACTS_INFO("Reference self-check: piecewise " << piecewise.size()
              << " state(s), continuous " << continuous.size());

    // A length mismatch has two very different causes and the counts alone do
    // not separate them:
    //   GRAZING   -- one recording touches a surface the other misses. The
    //                piecewise pass restarts on each surface, so it does not
    //                carry the continuous pass's accumulated drift and can
    //                catch a marginal surface that drifts past. Harmless: an
    //                unused reference state.
    //   REPEATED  -- the sequence lists one surface twice (a curler
    //                re-crossing a straw). The piecewise loop then records it
    //                twice while the navigator collapses it, leaving TWO
    //                reference states on one surface for DafReferenceFilter to
    //                match against. Not harmless.
    // So report which surfaces are unmatched, and compare the common ones
    // anyway rather than giving up on the whole check.
    auto duplicates = [](const DafReferenceTrajectory& t) {
      std::unordered_map<const Surface*, std::size_t> seen;
      for (const DafReferenceState& r : t) {
        ++seen[r.surface];
      }
      std::size_t extra = 0;
      for (const auto& entry : seen) {
        if (entry.second > 1) {
          extra += entry.second - 1;
        }
      }
      return extra;
    };
    const std::size_t repeatedPiecewise = duplicates(piecewise);
    const std::size_t repeatedContinuous = duplicates(continuous);
    if (repeatedPiecewise != 0 || repeatedContinuous != 0) {
      ACTS_INFO("  REPEATED SURFACES: piecewise " << repeatedPiecewise
                << ", continuous " << repeatedContinuous
                << " -- the sequence lists a surface more than once, so the "
                   "reference carries more than one state for it");
    }

    // Align the two by surface, in order, collecting what only one side has.
    std::vector<const Surface*> onlyPiecewise, onlyContinuous;
    double worstParams = 0., worstJacobian = 0., worstNoise = 0.;
    std::size_t worstIndex = 0, compared = 0, spanMismatched = 0;
    std::size_t i = 0, j = 0;
    // The first state after an insertion cannot be compared. `jacobian` is the
    // transport from the PREVIOUS reference state and `noise` the material over
    // the segment ENDING here (DafReferenceState.hpp), so if one recording has
    // an extra surface B between A and C, then at C it holds F(B->C) while the
    // other holds F(A->C). Different spans, so a large difference there is
    // arithmetic, not error -- reporting it as the worst case is misleading.
    bool afterInsertion = false;
    auto nextMatch = [](const DafReferenceTrajectory& t, std::size_t from,
                        const Surface* surface) {
      for (std::size_t k = from; k < t.size(); ++k) {
        if (t[k].surface == surface) {
          return k;
        }
      }
      return t.size();
    };
    while (i < piecewise.size() && j < continuous.size()) {
      if (piecewise[i].surface == continuous[j].surface) {
        if (afterInsertion) {
          afterInsertion = false;
          ++spanMismatched;
          ++i;
          ++j;
          continue;
        }
        const double dp = (piecewise[i].parameters - continuous[j].parameters)
                              .cwiseAbs().maxCoeff();
        const double dj = (piecewise[i].jacobian - continuous[j].jacobian)
                              .cwiseAbs().maxCoeff();
        const double dn =
            (piecewise[i].noise - continuous[j].noise).cwiseAbs().maxCoeff();
        if (dp + dj + dn > worstParams + worstJacobian + worstNoise) {
          worstIndex = i;
        }
        worstParams = std::max(worstParams, dp);
        worstJacobian = std::max(worstJacobian, dj);
        worstNoise = std::max(worstNoise, dn);
        ++i;
        ++j;
        ++compared;
        continue;
      }
      // Skip whichever side has to travel less far to resynchronise.
      const std::size_t inContinuous = nextMatch(continuous, j, piecewise[i].surface);
      const std::size_t inPiecewise = nextMatch(piecewise, i, continuous[j].surface);
      if (inContinuous != continuous.size() &&
          (inPiecewise == piecewise.size() ||
           inContinuous - j <= inPiecewise - i)) {
        for (; j < inContinuous; ++j) {
          onlyContinuous.push_back(continuous[j].surface);
        }
      } else if (inPiecewise != piecewise.size()) {
        for (; i < inPiecewise; ++i) {
          onlyPiecewise.push_back(piecewise[i].surface);
        }
      } else {
        onlyPiecewise.push_back(piecewise[i++].surface);
        onlyContinuous.push_back(continuous[j++].surface);
      }
      afterInsertion = true;
    }
    for (; i < piecewise.size(); ++i) {
      onlyPiecewise.push_back(piecewise[i].surface);
    }
    for (; j < continuous.size(); ++j) {
      onlyContinuous.push_back(continuous[j].surface);
    }

    auto listSurfaces = [](const std::vector<const Surface*>& surfaces) {
      std::ostringstream out;
      const std::size_t shown = std::min<std::size_t>(surfaces.size(), 6);
      for (std::size_t k = 0; k < shown; ++k) {
        out << " " << surfaces[k]->geometryId();
      }
      if (surfaces.size() > shown) {
        out << " ... (" << surfaces.size() - shown << " more)";
      }
      return out.str();
    };
    if (!onlyPiecewise.empty()) {
      ACTS_INFO("  only in PIECEWISE (" << onlyPiecewise.size() << "):"
                << listSurfaces(onlyPiecewise));
    }
    if (!onlyContinuous.empty()) {
      ACTS_INFO("  only in CONTINUOUS (" << onlyContinuous.size() << "):"
                << listSurfaces(onlyContinuous));
    }

    if (compared == 0) {
      ACTS_INFO("  no surface is common to both recordings");
      return;
    }
    if (spanMismatched != 0) {
      ACTS_INFO("  " << spanMismatched
                << " state(s) NOT compared: they follow an insertion, so the "
                   "two recordings transport them over different spans");
    }
    ACTS_INFO("  over " << compared << " common surface(s): max |dparameters| = "
              << worstParams << ", max |djacobian| = " << worstJacobian
              << ", max |dnoise| = " << worstNoise << " (worst at state "
              << worstIndex << ")");
    if (worstNoise > 1e-9) {
      ACTS_INFO("  NOISE DIFFERS: the piecewise recording is not accounting "
                "for the same material as the continuous one. Suspect a "
                "surface's post-update material being skipped by the "
                "propagation that starts on it.");
    }
  }

  /// Combine a forward-filtered and a backward-predicted state into the
  /// smoothed one, GenFit's calcAverageState (MeasuredStateOnPlane.cc:76):
  ///
  ///   C_s = (C_f^-1 + C_b^-1)^-1
  ///   x_s = C_s (C_f^-1 x_f + C_b^-1 x_b)
  ///
  /// The forward state must be FILTERED and the backward one PREDICTED, so the
  /// measurement on this surface is counted exactly once -- that pairing is
  /// what KalmanFitterInfo::getFittedState(biased=true) returns.
  ///
  /// @return false if the information sum is not invertible, leaving @p
  /// smoothed untouched so the caller can fall back.
  static bool combineForwardBackward(const BoundVector& forwardParams,
                                     const BoundSquareMatrix& forwardCov,
                                     const BoundVector& backwardParams,
                                     const BoundSquareMatrix& backwardCov,
                                     BoundVector& smoothedParams,
                                     BoundSquareMatrix& smoothedCov) {
    const Eigen::LLT<BoundSquareMatrix> fLLT(forwardCov);
    const Eigen::LLT<BoundSquareMatrix> bLLT(backwardCov);
    if (fLLT.info() != Eigen::Success || bLLT.info() != Eigen::Success) {
      return false;
    }
    const BoundSquareMatrix fInfo = forwardCov.inverse();
    const BoundSquareMatrix bInfo = backwardCov.inverse();

    // phi is periodic: combining 3.13 and -3.13 as plain numbers averages to
    // ~0, pointing the track backwards. Difference the backward state against
    // the forward one on the wrapped branch, combine the DIFFERENCE, and add it
    // back -- so the result is always within half a turn of the forward state.
    BoundVector delta = backwardParams - forwardParams;
    delta[eBoundPhi] = detail::radian_sym(delta[eBoundPhi]);

    const BoundSquareMatrix sum = fInfo + bInfo;
    const Eigen::LLT<BoundSquareMatrix> sLLT(sum);
    if (sLLT.info() != Eigen::Success) {
      return false;
    }
    smoothedCov = sum.inverse();
    if (!smoothedCov.allFinite()) {
      return false;
    }
    smoothedParams = forwardParams + smoothedCov * (bInfo * delta);
    // ACTS's canonical normalisation: folds theta back into [0, pi] and flips
    // phi by pi when it does, which a bare phi wrap would get wrong for a very
    // forward or backward track whose combined theta lands just outside.
    const auto [phi, theta] = detail::normalizePhiTheta(
        smoothedParams[eBoundPhi], smoothedParams[eBoundTheta]);
    smoothedParams[eBoundPhi] = phi;
    smoothedParams[eBoundTheta] = theta;
    return smoothedParams.allFinite();
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

    if (logger().doPrint(Logging::DEBUG)) {
      std::size_t nMultiCandidate = 0;
      for (const auto& [surface, group] : measurementGroups) {
        (void)surface;
        nMultiCandidate += static_cast<std::size_t>(group.size() > 1);
      }
      std::stringstream tempSs;
      for (double t : dafOptions.annealingConfig.setOfTemperatures) {
        tempSs << t << " ";
      }
      ACTS_DEBUG("DAF fit start: "
                 << measurementGroups.size() << " surface(s), "
                 << nMultiCandidate
                 << " with >1 candidate; temperature schedule: "
                 << tempSs.str() << "x "
                 << dafOptions.iterationsPerTemperature
                 << " iteration(s) each = " << totalIterations
                 << " passes; seed pos="
                 << sParameters.position(dafOptions.geoContext).transpose()
                 << " p=" << sParameters.absoluteMomentum()
                 << " pT=" << sParameters.transverseMomentum());
    }

    traj_t scratchCandidateStates;

    Acts::TrackContainer scratchTrackContainer{VectorTrackContainer{},
                                               traj_t{}};

    // Scratch for the two-filter smoother's backward filter, built the same way
    // as scratchTrackContainer. Owned here so it outlives every pass and is
    // cleared per use rather than reallocated.
    traj_t backwardCandidateStates;
    Acts::TrackContainer backwardTrackContainer{VectorTrackContainer{},
                                                traj_t{}};

    // Previous pass's smoothed state per surface, the linearisation points
    // referenceAnchorSmoothed re-references around. Empty on the first pass,
    // which is why that pass necessarily falls back to transporting the seed.
    std::unordered_map<const Surface*, BoundVector> referenceAnchors;
    // Scratch for the piecewise recording: one segment's worth of surfaces and
    // the states it produces, reused rather than reallocated per segment.
    std::vector<const Surface*> segmentSequence;
    traj_t segmentStates;
    DafReferenceTrajectory segmentTrajectory;

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

    // Every pass's smoothed chi2, in pass order -- the raw material for the
    // end-of-fit convergence summary.
    std::vector<double> chi2History;

    // Compact per-surface weight list, sorted by geometry id so the same
    // surface always appears in the same position across passes.
    auto weightSummary = [&]() {
      std::vector<std::pair<std::uint64_t, const DafCandidateState*>> sorted;
      sorted.reserve(candidateState.size());
      for (const auto& [surface, candState] : candidateState) {
        sorted.emplace_back(surface->geometryId().value(), &candState);
      }
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      std::stringstream ss;
      for (const auto& [geoIdValue, candState] : sorted) {
        ss << " " << GeometryIdentifier(geoIdValue).sensitive() << "=";
        if (candState->weights.size() == 1) {
          ss << candState->weights.front();
        } else {
          ss << "[";
          for (std::size_t i = 0; i < candState->weights.size(); ++i) {
            ss << (i > 0 ? "," : "") << candState->weights[i];
          }
          ss << "]";
        }
      }
      return ss.str();
    };

    // Snapshot the working candidate state into the caller's trace, if one was
    // asked for. Called once per pass: the state is overwritten in place
    // otherwise, so this copy is the only record of how the weights got to
    // where they ended up.
    auto recordTrace = [&](std::size_t iteration, bool isFinal) {
      if (dafOptions.candidateTrace == nullptr) {
        return;
      }
      DafCandidateTraceEntry entry;
      entry.iteration = iteration;
      entry.temperature =
          dafOptions.annealingConfig
              .setOfTemperatures[annealingState.currentTemperatureIndex];
      entry.isFinal = isFinal;
      entry.state = candidateState;
      dafOptions.candidateTrace->push_back(std::move(entry));
    };

    // The sequence each pass actually navigates. Starts as the caller's, and is
    // rebuilt from the pass's own re-seeded parameters when
    // DafFitterOptions::rebuildSequence is set -- see there for why. Owned here
    // so the pointer handed to make_propagator_options stays valid.
    std::vector<const Surface*> passSequence;
    if (sSequence != nullptr) {
      passSequence = *sSequence;
    }
    const std::vector<const Surface*>* passSequencePtr =
        sSequence != nullptr ? &passSequence : nullptr;

    // Rebuild passSequence for a pass, or leave it alone. Refuses a rebuilt
    // sequence that has lost a measurement-group surface: DafActor would then
    // never satisfy its termination condition and the pass would burn to the
    // path limit. Falling back costs this pass its re-linearisation; accepting
    // would cost the whole track.
    auto rebuildSequenceFor = [&](const BoundTrackParameters& seed,
                                  bool allowRebuild) {
      if (!allowRebuild || sSequence == nullptr ||
          !dafOptions.rebuildSequence) {
        return;
      }
      std::vector<const Surface*> rebuilt = dafOptions.rebuildSequence(seed);
      if (rebuilt.empty()) {
        ACTS_DEBUG("Sequence rebuild returned nothing; keeping the previous "
                   "pass's sequence");
        return;
      }
      const std::unordered_set<const Surface*> present(rebuilt.begin(),
                                                       rebuilt.end());
      for (const auto& [surface, group] : measurementGroups) {
        (void)group;
        if (!present.contains(surface)) {
          ACTS_DEBUG("Rebuilt sequence dropped measurement surface "
                     << surface->geometryId()
                     << "; keeping the previous pass's sequence");
          return;
        }
      }
      passSequence = std::move(rebuilt);
      ACTS_VERBOSE("Rebuilt the navigation sequence from this pass's seed: "
                   << passSequence.size() << " surfaces");
    };

    // Buffers for the reference-linearisation path, reused across passes.
    DafReferenceTrajectory referenceTrajectory;
    traj_t referenceRecordingStates;

    // The covariance every reference-linearised pass starts from. Deliberately
    // the ORIGINAL seed covariance and never the previous pass's smoothed one:
    // the reference carries everything learned so far, so the filter is free to
    // claim no information at all, and re-filtering the same measurements
    // against a covariance that already contains them would count them twice
    // (the reason reseedCovarianceScaling exists on the running-state path).
    BoundSquareMatrix referenceStartCovariance = BoundSquareMatrix::Identity();
    if (sParameters.covariance().has_value()) {
      referenceStartCovariance = sParameters.covariance().value();
    } else if (dafOptions.referenceLinearization) {
      ACTS_DEBUG("Seed carries no covariance; the reference filter starts from "
                 "the identity");
    }
    // The same treatment the reseed path gets: reset off-diagonals, scale,
    // then clip per parameter. A bare multiply would drive sigma(phi) and
    // sigma(theta) toward the branch cut as soon as the scaling is turned up.
    blowUpCovariance(referenceStartCovariance,
                     dafOptions.referenceStartCovarianceScaling,
                     dafOptions.referenceStartResetOffDiagonals,
                     dafOptions.referenceStartVarianceCeiling,
                     dafOptions.referenceStartVarianceCeilings);

    // Which surface referenceStartCovariance is valid on, and whether it has
    // been replaced by the transported version cached after the first pass.
    //
    // Pass 1 propagates the reference from the seed surface, so the filter
    // transports the seed covariance out to the first recorded surface itself.
    // Every later pass starts the propagation AT the first measurement surface
    // -- that is where the re-seeded parameters live -- so its first recorded
    // state has J = I and N = 0 and no transport happens. Handing it the raw
    // seed covariance would apply a perigee covariance at a straw, where a
    // clipped var_d0 of 0.04 mm^2 is tighter than the measurement itself
    // (0.17 mm^2) and pins the track to the seed's d0.
    //
    // The right prior there is the one pass 1 already computed: the predicted
    // covariance at its first measurement state, which is the seed information
    // transported to that surface plus accumulated noise, with no measurement
    // absorbed yet. Cache it and reuse it.
    const Surface* referenceStartSurface = nullptr;
    BoundSquareMatrix cachedStartCovariance = BoundSquareMatrix::Identity();
    bool referenceStartCovarianceCached = false;

    // One forward pass into @p container: either the default running-state
    // filter (propagate and update as it goes, GenFit's plain KalmanFitter) or
    // the reference-linearised split (record, then filter without propagating,
    // GenFit's KalmanFitterRefTrack). Everything downstream -- smoothing, the
    // weight update, salvaging -- is identical either way.
    auto runForwardPass =
        [&](const BoundTrackParameters& seed, bool firstPass, auto& container)
        -> Result<typename std::decay_t<decltype(container)>::TrackProxy> {
      rebuildSequenceFor(seed, !firstPass);
      scratchCandidateStates.clear();

      if (!dafOptions.referenceLinearization) {
        auto propagatorOptions = make_propagator_options(
            measurementGroups, dafOptions, passSequencePtr, candidateState,
            scratchCandidateStates, firstPass);
        return filter_impl(seed, propagatorOptions, container);
      }

      // ================= x_ref, F and Q ARE REBUILT HERE =================
      // Once per annealing pass, and nowhere else. Everything downstream in
      // this pass -- filter, smoother, weight update -- reads the values
      // recorded by the single propagation below and never recomputes them.
      ACTS_DEBUG("=== REFERENCE REBUILD: discarding "
                 << referenceTrajectory.size()
                 << " state(s) from the previous pass and re-recording x_ref, "
                    "F and Q by one RK propagation from "
                 << seed.referenceSurface().geometryId()
                 << "  loc0=" << seed.parameters()[eBoundLoc0]
                 << " loc1=" << seed.parameters()[eBoundLoc1]
                 << " phi=" << seed.parameters()[eBoundPhi]
                 << " theta=" << seed.parameters()[eBoundTheta]
                 << " q/p=" << seed.parameters()[eBoundQOverP]
                 << "  (source: "
                 << (firstPass ? "the track-finding seed"
                               : "the PREVIOUS pass's smoothed state at its "
                                 "first measurement, covariance blown up")
                 << ")");
      referenceTrajectory.clear();
      referenceRecordingStates.clear();

      // The reference propagation carries a ZERO covariance, so the stepping
      // covariance at each surface is the accumulated process noise alone --
      // that is what makes DafActor's N = C_k - F*C_k-1*F^T well-conditioned.
      // The filter's covariance comes from referenceStartCovariance instead.
      const BoundSquareMatrix zeroCovariance = BoundSquareMatrix::Zero();
      BoundTrackParameters referenceSeed(
          seed.referenceSurface().getSharedPtr(), seed.parameters(),
          zeroCovariance, seed.particleHypothesis());

      auto propagatorOptions = make_propagator_options(
          measurementGroups, dafOptions, passSequencePtr, candidateState,
          scratchCandidateStates, firstPass, /*recordReference=*/true,
          &referenceTrajectory);

      // One segment per step of the prepared sequence, each propagated from the
      // previous reference state -- GenFit's prepareTrack. The boundaries fall
      // where the continuous recording already divides the noise, so with
      // anchoring off this must reproduce it; referencePiecewiseSelfCheck is
      // what verifies that.
      auto recordPiecewise = [&]() -> Result<void> {
        if (passSequencePtr == nullptr || passSequencePtr->size() < 2) {
          return Result<void>::success();   // nothing to split
        }
        BoundVector current = seed.parameters();
        const Surface* currentSurface = &seed.referenceSurface();
        std::size_t skippedTargets = 0;
        const BoundVector* previousReference = nullptr;
        BoundVector previousReferenceStorage = BoundVector::Zero();

        // Built ONCE, outside the loop. make_propagator_options deep-copies the
        // whole measurement-group map into the actor; at ~110 segments x ~9
        // passes that is ~1000 copies of a ~50-entry map per track, where the
        // continuous path pays for 9. Nothing in the options varies per segment
        // except the two-surface sequence, and the actor holds the trajectory by
        // POINTER, so clearing and refilling segmentTrajectory keeps working.
        auto segmentOptions = make_propagator_options(
            measurementGroups, dafOptions, &segmentSequence, candidateState,
            scratchCandidateStates, firstPass, /*recordReference=*/true,
            &segmentTrajectory);
        // A segment spans ONE step of the sequence -- millimetres to a few
        // centimetres. A segment that never intersects its target instead runs
        // past endOfSurfaces() and keeps stepping to the global limit (50000
        // via ACTS_MAX_STEPS), and the DEBUG log shows ~12 such targets per
        // pass, so that waste is ~100x the useful work. Cap it well above any
        // real segment and far below the global limit.
        segmentOptions.maxSteps =
            std::min<unsigned int>(segmentOptions.maxSteps, 2000u);

        for (std::size_t i = 0; i < passSequencePtr->size(); ++i) {
          const Surface* target = (*passSequencePtr)[i];
          segmentSequence.clear();
          if (currentSurface != target) {
            segmentSequence.push_back(currentSurface);
          }
          segmentSequence.push_back(target);

          segmentTrajectory.clear();
          segmentStates.clear();
          // externalSurfaces is a COPY made by make_propagator_options, so the
          // per-segment sequence has to be pushed in here rather than relied on
          // through segmentSequence. Two pointers.
          segmentOptions.navigation.externalSurfaces = segmentSequence;
          BoundTrackParameters segmentSeed(
              currentSurface->getSharedPtr(), current, zeroCovariance,
              seed.particleHypothesis());
          // Where this segment starts. A start that has left the detector --
          // or a momentum that has run away -- is what sends the stepper off
          // the field map, and it points at the hand-off below rather than at
          // the propagation itself.
          const Vector3 segmentStart =
              segmentSeed.position(dafOptions.geoContext);
          ACTS_DEBUG("Piecewise segment " << i << "/" << passSequencePtr->size()
                     << ": " << currentSurface->geometryId() << " -> "
                     << target->geometryId() << "  start r="
                     << segmentStart.head<2>().norm()
                     << " z=" << segmentStart.z()
                     << " loc0=" << current[eBoundLoc0]
                     << " loc1=" << current[eBoundLoc1]
                     << " phi=" << current[eBoundPhi]
                     << " theta=" << current[eBoundTheta]
                     << " q/p=" << current[eBoundQOverP]
                     << " p=" << segmentSeed.absoluteMomentum());
          auto segmentResult =
              record_impl(segmentSeed, segmentOptions, segmentStates);
          {
            std::ostringstream recorded;
            for (const DafReferenceState& r : segmentTrajectory) {
              recorded << " " << r.surface->geometryId()
                       << (r.hasMeasurement ? "(m)" : "");
            }
            ACTS_DEBUG("  segment " << i << " -> "
                       << (segmentResult.ok()
                               ? std::string("ok")
                               : segmentResult.error().message())
                       << ", recorded " << segmentTrajectory.size()
                       << " state(s):"
                       << (segmentTrajectory.empty() ? std::string(" none")
                                                     : recorded.str()));
          }
          // Whether the segment produced this target's state is the ONLY
          // criterion. record_impl's return code cannot be used as one: it
          // ends with a guard demanding at least one measurement state, which
          // is written for a whole-track recording, and a segment is ONE step
          // of the sequence that usually lands on a material-only surface. So
          // NoMeasurementFound is the normal case here, and by the time that
          // guard runs the actor has already appended what the segment owed.
          auto it = std::find_if(segmentTrajectory.rbegin(),
                                 segmentTrajectory.rend(),
                                 [&](const DafReferenceState& r) {
                                   return r.surface == target;
                                 });
          if (it == segmentTrajectory.rend()) {
            // NOT the end of the track. The DirectNavigator drops sequence
            // entries the trajectory does not intersect and carries on -- the
            // continuous recording of the same track skips them too, which is
            // why it is SHORTER than the sequence. Do the same: leave the
            // state where it is and aim at the next entry. Treating this as
            // the end truncated the reference to 40 of 87 steps.
            //
            // A propagation error here is a CONSEQUENCE of the miss, not an
            // independent failure: a two-surface segment that never intersects
            // its target runs past endOfSurfaces() and keeps stepping until it
            // hits the path limit or leaves the field map. `current` is
            // untouched, so skipping is safe.
            ++skippedTargets;
            ACTS_DEBUG("Piecewise reference: segment " << i << " never reached "
                       << target->geometryId() << " (from "
                       << currentSurface->geometryId() << " at r="
                       << segmentStart.head<2>().norm() << " z="
                       << segmentStart.z() << ")"
                       << (segmentResult.ok()
                               ? std::string()
                               : " -- propagation also reported " +
                                     segmentResult.error().message())
                       << "; skipping it, as the continuous navigator does");
            continue;
          }
          DafReferenceState record = *it;

          // Re-linearise around the previous pass's smoothed state where there
          // is one. F and N stay the transport of the PREVIOUS reference to
          // this surface; only the expansion point moves, and deltaState is
          // what absorbs the difference.
          if (dafOptions.referenceAnchorSmoothed) {
            auto anchor = referenceAnchors.find(target);
            if (anchor != referenceAnchors.end()) {
              record.parameters = anchor->second;
            }
          }
          if (previousReference != nullptr) {
            record.deltaState =
                record.parameters - record.jacobian * (*previousReference);
          }
          referenceTrajectory.push_back(record);

          previousReferenceStorage = record.parameters;
          previousReference = &previousReferenceStorage;
          current = record.parameters;
          currentSurface = target;
        }
        std::size_t withMeasurement = 0;
        for (const DafReferenceState& r : referenceTrajectory) {
          withMeasurement += static_cast<std::size_t>(r.hasMeasurement);
        }
        ACTS_DEBUG("Piecewise reference: " << referenceTrajectory.size()
                   << " state(s) from " << passSequencePtr->size()
                   << " sequence step(s), " << withMeasurement
                   << " with a measurement, " << skippedTargets
                   << " unreachable surface(s) skipped");
        return Result<void>::success();
      };

      if (dafOptions.referencePiecewise) {
        auto piecewiseResult = recordPiecewise();
        if (!piecewiseResult.ok()) {
          return piecewiseResult.error();
        }
        if (dafOptions.referencePiecewiseSelfCheck) {
          DafReferenceTrajectory continuousTrajectory;
          traj_t continuousStates;
          auto checkOptions = make_propagator_options(
              measurementGroups, dafOptions, passSequencePtr, candidateState,
              scratchCandidateStates, firstPass, /*recordReference=*/true,
              &continuousTrajectory);
          auto checkResult =
              record_impl(referenceSeed, checkOptions, continuousStates);
          if (checkResult.ok()) {
            reportReferenceDifference(referenceTrajectory, continuousTrajectory,
                                      logger());
          }
        }
      } else {
        auto recordResult = record_impl(referenceSeed, propagatorOptions,
                                        referenceRecordingStates);
        if (!recordResult.ok()) {
          return recordResult.error();
        }
      }

      if (logger().doPrint(Logging::DEBUG)) {
        std::size_t withMeas = 0;
        for (const DafReferenceState& r : referenceTrajectory) {
          withMeas += static_cast<std::size_t>(r.hasMeasurement);
        }
        ACTS_DEBUG("=== REFERENCE REBUILD done: "
                   << referenceTrajectory.size() << " state(s), " << withMeas
                   << " with a measurement, s = 0 -> "
                   << (referenceTrajectory.empty()
                           ? 0.
                           : referenceTrajectory.back().pathLength)
                   << " mm. x_ref, F and Q are now FIXED for this pass; the "
                      "filter and smoother below do pure linear algebra on "
                      "them and touch neither the propagator nor the geometry.");
      }

      // Only when this pass's first recorded state really is that surface;
      // a rebuilt sequence could have moved it, and then the transported prior
      // does not belong here.
      const bool useCached =
          referenceStartCovarianceCached && !referenceTrajectory.empty() &&
          referenceTrajectory.front().surface == referenceStartSurface;
      if (referenceStartCovarianceCached && !useCached) {
        ACTS_DEBUG("Reference pass starts on a different surface than the "
                   "cached prior was computed for; falling back to the seed "
                   "covariance");
      }

      // Which prior THIS pass filters against, and the scaling already folded
      // into it. Both candidates are scaled at the point they are built
      // (referenceStartCovariance and cachedStartCovariance), so whichever is
      // chosen here carries the scaling -- it applies on every pass, not only
      // the first. Printed per pass so that is verifiable rather than assumed.
      const BoundSquareMatrix& startPrior =
          useCached ? cachedStartCovariance : referenceStartCovariance;
      ACTS_DEBUG("=== REFERENCE PRIOR for this pass: "
                 << (useCached ? "the cached transported prior"
                               : "the seed covariance")
                 << ", scaled by " << dafOptions.referenceStartCovarianceScaling
                 << " -> tr = " << startPrior.trace()
                 << ", sqrt(diag) loc0=" << std::sqrt(std::max(0., startPrior(eBoundLoc0, eBoundLoc0)))
                 << " loc1=" << std::sqrt(std::max(0., startPrior(eBoundLoc1, eBoundLoc1)))
                 << " phi=" << std::sqrt(std::max(0., startPrior(eBoundPhi, eBoundPhi)))
                 << " theta=" << std::sqrt(std::max(0., startPrior(eBoundTheta, eBoundTheta)))
                 << " qop=" << std::sqrt(std::max(0., startPrior(eBoundQOverP, eBoundQOverP))));

      return detail::dafReferenceFilter(
          referenceTrajectory, scratchCandidateStates, startPrior,
          candidateState, dafOptions, container, logger());
    };

    // One-line convergence story: chi2 of every pass, with '|' marking the
    // annealing temperature steps, plus the weights the fit ended on.
    auto logFitSummary = [&](const char* outcome, double finalChi2) {
      if (!logger().doPrint(Logging::DEBUG)) {
        return;
      }
      chi2History.push_back(finalChi2);
      std::stringstream ss;
      for (std::size_t i = 0; i < chi2History.size(); ++i) {
        if (i > 0 && i % dafOptions.iterationsPerTemperature == 0) {
          ss << " |";
        }
        ss << " " << chi2History[i];
      }
      ACTS_DEBUG("DAF " << outcome << " after " << chi2History.size() << "/"
                        << totalIterations << " passes; chi2 per pass:"
                        << ss.str() << "; final w:" << weightSummary());
    };

    // GenFit's smoothing: a second Kalman filter run BACKWARD from the last
    // measurement, combined with the forward one. Returns false if it could not
    // be done, and the caller then falls back to the RTS smoother -- this is
    // strictly an alternative route to the same smoothed estimate, so failing
    // over costs accuracy only through non-linearity, never correctness.
    auto smoothByTwoFilter = [&](auto& track, auto& container) -> bool {
      if (!dafOptions.useTwoFilterSmoothing) {
        return false;
      }

      // The backward filter starts where the forward one ended, with its
      // information thrown away -- GenFit blows up the last point's backward
      // prediction before running (KalmanFitterRefTrack.cc:202).
      auto lastMeasurement = findLastMeasurementState(track);
      if (!lastMeasurement.ok()) {
        ACTS_DEBUG("Two-filter smoothing: no last measurement state to start "
                   "the backward filter from; falling back to RTS");
        return false;
      }
      const auto lastState = lastMeasurement.value();
      if (!lastState.hasFiltered()) {
        ACTS_DEBUG("Two-filter smoothing: the last measurement carries no "
                   "filtered state; falling back to RTS");
        return false;
      }
      BoundSquareMatrix startCov = lastState.filteredCovariance();
      blowUpCovariance(startCov, dafOptions.twoFilterBlowUpFactor,
                       dafOptions.reseedResetOffDiagonals,
                       dafOptions.reseedVarianceCeiling,
                       dafOptions.reseedVarianceCeilings);
      BoundTrackParameters backwardSeed(
          lastState.referenceSurface().getSharedPtr(), lastState.filtered(),
          startCov, currentSeed.particleHypothesis());

      // The SAME sequence, not a reversed copy. DirectNavigator::nextSurface
      // decrements its index when the propagation direction is backward
      // (DirectNavigator.hpp:167), so it walks this list inward on its own --
      // reversing it here as well would double-reverse and walk it outward
      // again, silently. ACTS's own KalmanFitter passes its sequence unchanged
      // to the reversed pass for the same reason.
      // firstPass=false: the straw prior is a first-pass-only initialisation
      // and must not be re-applied here, and this pass must not touch the
      // candidate weights at all -- it only re-filters the combination the
      // current weights already define.
      backwardCandidateStates.clear();
      auto backwardOptions = make_propagator_options(
          measurementGroups, dafOptions, passSequencePtr,
          candidateState, backwardCandidateStates, /*firstPass=*/false,
          /*recordReference=*/false, /*referenceTrajectory=*/nullptr,
          Direction::Backward());

      backwardTrackContainer.clear();
      auto backwardResult =
          filter_impl(backwardSeed, backwardOptions, backwardTrackContainer);
      if (!backwardResult.ok()) {
        ACTS_DEBUG("Two-filter smoothing: backward pass failed ("
                   << backwardResult.error() << "); falling back to RTS");
        return false;
      }

      // Index the backward pass by surface so each forward state can find its
      // partner in one lookup.
      std::unordered_map<const Surface*, std::size_t> backwardBySurface;
      for (const auto ts : backwardResult.value().trackStatesReversed()) {
        if (ts.hasPredicted()) {
          backwardBySurface.emplace(&ts.referenceSurface(), ts.index());
        }
      }

      std::size_t combined = 0;
      std::size_t missing = 0;
      // Reversed, not forward: forward iteration needs a stemIndex, which the
      // DAF's tracks never set (the propagation only links tip-ward), and
      // asking for it throws "Track has no stem index". Order is irrelevant
      // here anyway -- each surface is combined independently of the others.
      for (auto ts : track.trackStatesReversed()) {
        if (!ts.hasCalibrated() || !ts.hasFiltered()) {
          continue;
        }
        auto it = backwardBySurface.find(&ts.referenceSurface());
        if (it == backwardBySurface.end()) {
          ++missing;
          continue;
        }
        const auto back =
            backwardTrackContainer.trackStateContainer().getTrackState(
                it->second);
        BoundVector smoothedParams;
        BoundSquareMatrix smoothedCov;
        if (!combineForwardBackward(ts.filtered(), ts.filteredCovariance(),
                                    back.predicted(), back.predictedCovariance(),
                                    smoothedParams, smoothedCov)) {
          ++missing;
          continue;
        }
        // DafActor allocates each state with Predicted|Filtered|Jacobian|
        // Calibrated and NOT Smoothed, so the component has to be added before
        // it can be written -- GainMatrixSmoother.hpp:85 does the same before
        // its own write, which is why the RTS path works without this.
        ts.addComponents(TrackStatePropMask::Smoothed);
        ts.smoothed() = smoothedParams;
        ts.smoothedCovariance() = smoothedCov;
        ++combined;
      }

      if (combined == 0) {
        ACTS_DEBUG("Two-filter smoothing: no surface could be combined; "
                   "falling back to RTS");
        return false;
      }
      if (missing > 0) {
        // A partial result would leave some surfaces holding whatever was in
        // `smoothed` before, which the weight update would then score against.
        // Redoing the whole track with the RTS smoother is the only consistent
        // option.
        ACTS_DEBUG("Two-filter smoothing: " << missing << " of "
                   << (combined + missing)
                   << " surfaces had no usable backward partner; falling back "
                      "to RTS for the whole track");
        return false;
      }
      ACTS_VERBOSE("Two-filter smoothing: combined " << combined
                                                     << " surface(s)");
      return true;
    };

    // Runs one forward+smooth pass into the caller's output container using
    // the given seed and whatever weights candidateState currently holds, then
    // extrapolates to the reference surface. This is the "official" result-
    // producing pass, shared by the normal final iteration and by salvaging.
    auto finalizeInto = [&](const BoundTrackParameters& seed,
                            bool firstPass) -> Result<TrackProxy> {
      // Not on the first pass: there the seed IS the caller's, so the rebuild
      // would reproduce the sequence it was already handed.
      auto forwardResult = runForwardPass(seed, firstPass, trackContainer);
      if (!forwardResult.ok()) {
        return forwardResult.error();
      }
      TrackProxy track = forwardResult.value();

      // GenFit's two-filter smoothing when asked for, the RTS smoother
      // otherwise -- and also whenever the backward route could not complete,
      // since a partially combined track would leave some surfaces holding a
      // stale smoothed state for the weight update to score against.
      if (!smoothByTwoFilter(track, trackContainer)) {
        auto smoothRes = dafOptions.extensions.smoother(
            dafOptions.geoContext, trackContainer.trackStateContainer(),
            track.tipIndex(), logger());
        if (!smoothRes.ok()) {
          return smoothRes.error();
        }
      }

      if (trackContainer.hasColumn(hashString("smoothed"))) {
        track.template component<bool, hashString("smoothed")>() = true;
      }

      if (!track.hasReferenceSurface() &&
          dafOptions.referenceSurface != nullptr) {
        // Same reasoning as the direct-navigation branch of
        // ActsExamples::DafFitterFunction: this extrapolation crosses whatever
        // lies between the innermost measurement and the reference surface, and
        // without a MaterialInteractor none of that material enters the
        // transported covariance. Follows the fit's own scattering/energy-loss
        // switches so a no-material fit stays a no-material fit.
        typename propagator_t::template Options<ActorList<MaterialInteractor>>
            extrapolationOptions(dafOptions.geoContext,
                                 dafOptions.magFieldContext);
        auto& matInteractor =
            extrapolationOptions.actorList.template get<MaterialInteractor>();
        matInteractor.multipleScattering = dafOptions.multipleScattering;
        matInteractor.energyLoss = dafOptions.energyLoss;
        auto extrapolationResult = extrapolateTrackToReferenceSurface(
            track, *dafOptions.referenceSurface, m_propagator,
            extrapolationOptions, dafOptions.referenceSurfaceStrategy,
            logger());
        if (!extrapolationResult.ok()) {
          return extrapolationResult.error();
        }
      }

      // Export the weights this (result-producing) pass consumed -- also
      // correct for the salvage path, which restores candidateState to the
      // last good pass's weights before replaying through here.
      if (dafOptions.finalCandidateState != nullptr) {
        *dafOptions.finalCandidateState = candidateState;
      }
      // Close the trace with the same weights, flagged as the pass whose
      // output the caller actually got. On the salvage path this is the
      // restored last-good state, so the trace stays consistent with the
      // returned track rather than ending on the failed pass.
      recordTrace(chi2History.size(), /*isFinal=*/true);

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
      auto salvaged = finalizeInto(*lastGoodSeed, false);
      if (!salvaged.ok()) {
        ACTS_DEBUG("Salvage pass unexpectedly failed: " << salvaged.error());
        return salvaged.error();
      }
      logFitSummary("salvaged", salvaged.value().chi2());
      return salvaged;
    };

    // Set once a convergence test fires, so the NEXT pass is the one that
    // writes the output and the loop then ends -- GenFit's oneLastIter, which
    // gives the converged weights one full pass to act through rather than
    // returning the pass that merely detected convergence.
    bool oneLastIteration = false;
    // GenFit seeds lastPval to -1 so that its `lastPval >= minPval_` term is
    // false on the first evaluation: the p-value test needs TWO consecutive
    // passes, and this is what stops it firing on the strength of one.
    double lastPValue = -1.;

    for (std::size_t iter = 0; iter < totalIterations; ++iter) {
      const bool isFinalIteration =
          oneLastIteration || (iter + 1 == totalIterations);

      const auto& seedPars = currentSeed.parameters();
      ACTS_DEBUG("DAF pass "
                 << (iter + 1) << "/" << totalIterations << " at T="
                 << dafOptions.annealingConfig
                        .setOfTemperatures[annealingState.currentTemperatureIndex]
                 << (isFinalIteration
                         ? " (final pass, writes the output track)"
                         : "")
                 << "; seed loc=(" << seedPars[eBoundLoc0] << ", "
                 << seedPars[eBoundLoc1] << ") phi=" << seedPars[eBoundPhi]
                 << " theta=" << seedPars[eBoundTheta]
                 << " q/p=" << seedPars[eBoundQOverP]
                 << " p=" << currentSeed.absoluteMomentum()
                 << " pT=" << currentSeed.transverseMomentum());

      if (isFinalIteration) {
        auto finalTrack = finalizeInto(currentSeed, iter == 0);
        if (finalTrack.ok()) {
          logFitSummary("finished", finalTrack.value().chi2());
          return finalTrack.value();
        }
        ACTS_DEBUG("Final DAF pass failed: " << finalTrack.error());
        return salvage(finalTrack.error());
      }

      scratchTrackContainer.clear();
      auto forwardResult =
          runForwardPass(currentSeed, iter == 0, scratchTrackContainer);
      if (!forwardResult.ok()) {
        ACTS_DEBUG("DAF pass " << iter << " failed: " << forwardResult.error());
        return salvage(forwardResult.error());
      }
      auto forwardTrack = forwardResult.value();

      if (!smoothByTwoFilter(forwardTrack, scratchTrackContainer)) {
        auto smoothRes = dafOptions.extensions.smoother(
            dafOptions.geoContext, scratchTrackContainer.trackStateContainer(),
            forwardTrack.tipIndex(), logger());
        if (!smoothRes.ok()) {
          ACTS_DEBUG("Smoothing step failed on pass " << iter << ": "
                                                      << smoothRes.error());
          return salvage(smoothRes.error());
        }
      }

      // Cache the prior this pass had at its innermost measurement, for every
      // later pass to start from (see referenceStartSurface). Taken after
      // smoothing only for convenience -- the smoother writes the smoothed
      // components and leaves the predicted ones alone. trackStatesReversed
      // runs tip-first, so the last measurement state seen is the innermost.
      if (dafOptions.referenceLinearization && !referenceStartCovarianceCached) {
        for (const auto ts : forwardTrack.trackStatesReversed()) {
          if (ts.hasCalibrated()) {
            cachedStartCovariance = ts.predictedCovariance();
            referenceStartSurface = &ts.referenceSurface();
          }
        }
        // Scaled the same as referenceStartCovariance, or the option would
        // only ever reach pass 1: passes 2..N take this cached prior instead,
        // and they are where the fit is actually decided. The running-state
        // path inflates its re-seed covariance on EVERY pass
        // (reseedCovarianceScaling), so leaving this one unscaled is also what
        // made the two paths incomparable.
        blowUpCovariance(cachedStartCovariance,
                         dafOptions.referenceStartCovarianceScaling,
                         dafOptions.referenceStartResetOffDiagonals,
                         dafOptions.referenceStartVarianceCeiling,
                         dafOptions.referenceStartVarianceCeilings);
        if (referenceStartSurface != nullptr) {
          referenceStartCovarianceCached = true;
          ACTS_DEBUG("=== REFERENCE PRIOR cached at "
                     << referenceStartSurface->geometryId()
                     << ": every LATER pass filters its rebuilt reference "
                        "against this covariance instead of the seed's. Only "
                        "captured once, from this pass.");
          ACTS_DEBUG("Cached the reference filter's prior at surface "
                     << referenceStartSurface->geometryId()
                     << " for the remaining passes");
        }
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
        // Keep the smoothed parameters -- that is the re-linearisation this
        // re-seeding exists for -- but inflate the covariance that comes with
        // them. It already contains every measurement on the track, and the
        // next pass is about to filter those same measurements again; carrying
        // it over unchanged counts their information twice, compounding once
        // per pass. See DafFitterOptions::reseedCovarianceScaling.
        BoundSquareMatrix seedCov = fm.smoothedCovariance();

        // GenFit's MeasuredStateOnPlane::blowUpCov, in its own order: reset
        // the off-diagonals, scale, then clip every variance to a ceiling (its
        // blowUpMaxVal, 1e6).
        //
        // The clip is what makes a large scaling survivable, and why GenFit can
        // use 1000 where a uniform ACTS scaling cannot. GenFit's plane state
        // (q/p, u', v', u, v) is entirely unbounded, so a huge variance just
        // means "no information". An ACTS bound state carries phi and theta,
        // which are periodic: past sigma ~ pi the prior is not merely flat but
        // meaningless, the first update sends the direction anywhere, and
        // normalizePhiTheta folds it back with a phi flip. Ceilings are per
        // parameter so loc0/loc1/qop can take the full blow-up while the angles
        // stay well inside the branch cut.
        blowUpCovariance(seedCov, dafOptions.reseedCovarianceScaling,
                         dafOptions.reseedResetOffDiagonals,
                         dafOptions.reseedVarianceCeiling,
                         dafOptions.reseedVarianceCeilings);
        const BoundVector previousSeed = currentSeed.parameters();
        currentSeed = BoundTrackParameters(
            fm.referenceSurface().getSharedPtr(), fm.smoothed(), seedCov,
            currentSeed.particleHypothesis());
        // This assignment is the ONLY thing that makes the next pass's x_ref
        // differ from this one's. Nothing else feeds the rebuild above.
        ACTS_DEBUG("=== REFERENCE SEED for the next pass: taken from the "
                   "smoothed state at "
                   << fm.referenceSurface().geometryId()
                   << ", covariance blown up by "
                   << dafOptions.reseedCovarianceScaling
                   << ".  q/p " << previousSeed[eBoundQOverP] << " -> "
                   << currentSeed.parameters()[eBoundQOverP]
                   << ", phi " << previousSeed[eBoundPhi] << " -> "
                   << currentSeed.parameters()[eBoundPhi]
                   << ", theta " << previousSeed[eBoundTheta] << " -> "
                   << currentSeed.parameters()[eBoundTheta]);
        ACTS_VERBOSE("Re-seeding next pass from the smoothed first-measurement "
                     "state on surface "
                     << fm.referenceSurface().geometryId()
                     << ": params=" << currentSeed.parameters().transpose());
      } else {
        ACTS_DEBUG("Could not find first measurement state to re-seed next "
                   "pass, reusing previous seed: "
                   << firstMeasurement.error());
      }

      // This pass's smoothed states become the next pass's linearisation
      // points. Captured from the same trajectory the weight update scores
      // against, before the re-seed below moves anything.
      if (dafOptions.referenceAnchorSmoothed) {
        referenceAnchors.clear();
        for (const auto ts : forwardTrack.trackStatesReversed()) {
          if (ts.hasSmoothed()) {
            referenceAnchors[&ts.referenceSurface()] = ts.smoothed();
          }
        }
      }

      const DafPassSummary passSummary = updateCandidateWeights(
          forwardTrack, measurementGroups, dafOptions, candidateState,
          annealing, annealingState, scratchCandidateStates,
          dafOptions.referenceLinearization ? &referenceTrajectory : nullptr);

      // GenFit's convergence tests, in GenFit's order and with its structure
      // (DAF::processTrackWithRep). The weight test is primary; the p-value
      // test is a FALLBACK that can only upgrade a "not converged" verdict,
      // never overturn a converged one.
      bool converged =
          dafOptions.weightConvergenceTolerance > 0. &&
          passSummary.maxAbsWeightChange <
              dafOptions.weightConvergenceTolerance;
      if (converged) {
        ACTS_DEBUG("DAF weights converged after pass "
                   << (iter + 1) << " (max |dw| = "
                   << passSummary.maxAbsWeightChange << " < "
                   << dafOptions.weightConvergenceTolerance << ")");
      }

      const double pValue =
          (passSummary.ndf > 0.)
              ? detail::chi2SurvivalFunction(passSummary.weightedChi2,
                                             passSummary.ndf)
              : std::numeric_limits<double>::quiet_NaN();
      if (!converged && dafOptions.pValueConvergenceTolerance > 0. &&
          iter + 1 >= dafOptions.pValueMinIterations &&
          std::isfinite(pValue) && pValue > dafOptions.pValueMinimum &&
          lastPValue >= dafOptions.pValueMinimum &&
          std::abs(lastPValue - pValue) <
              dafOptions.pValueConvergenceTolerance) {
        ACTS_DEBUG("DAF converged by p-value after pass "
                   << (iter + 1) << " (p = " << pValue << ", was " << lastPValue
                   << ") even though the weights are still moving (max |dw| = "
                   << passSummary.maxAbsWeightChange << ")");
        converged = true;
      }
      lastPValue = pValue;

      // minIterations gates BOTH tests, as GenFit's `iBeta >= minIterations_-1
      // && converged` does -- the p-value route has its own, separate floor
      // already applied above.
      if (converged && iter + 1 >= dafOptions.minIterations) {
        oneLastIteration = true;
      }

      // Record before the annealing step below, so the entry carries the
      // temperature this pass actually ran at rather than the next one's.
      recordTrace(iter, /*isFinal=*/false);

      if (logger().doPrint(Logging::DEBUG)) {
        std::stringstream msg;
        msg << "DAF pass " << (iter + 1) << "/" << totalIterations
            << " done: chi2 ";
        if (!chi2History.empty()) {
          msg << chi2History.back() << " -> ";
        }
        msg << forwardTrack.chi2() << "; meas="
            << forwardTrack.nMeasurements() << "/" << measurementGroups.size()
            << " holes=" << forwardTrack.nHoles() << "; w:" << weightSummary();
        ACTS_DEBUG(msg.str());
      }
      chi2History.push_back(forwardTrack.chi2());

      if ((iter + 1) % dafOptions.iterationsPerTemperature == 0) {
        const auto& temperatures =
            dafOptions.annealingConfig.setOfTemperatures;
        const double oldT =
            temperatures[annealingState.currentTemperatureIndex];
        annealing.anneal(annealingState);
        const double newT =
            temperatures[annealingState.currentTemperatureIndex];
        ACTS_DEBUG("DAF annealing step after pass "
                   << (iter + 1) << ": T " << oldT << " -> " << newT
                   << " (temperature "
                   << annealingState.currentTemperatureIndex + 1 << "/"
                   << temperatures.size() << ")"
                   << (annealingState.equilibriumReached
                           ? ", equilibrium reached"
                           : ""));
      }
    }

    return DafFitterError::AnnealingDidNotConverge;
  }
};

/// @}

}  // namespace Acts

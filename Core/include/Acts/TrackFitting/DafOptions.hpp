// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/SourceLink.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/PropagatorOptions.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilterExtensions.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/detail/VoidFitterComponents.hpp"
#include "Acts/Utilities/AnnealingUtility.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/Delegate.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace Acts {

/// @addtogroup track_fitting
/// @{

/// The set of source links competing as hit hypotheses for a single surface.
/// Reuses CKF's small-vector cap since both CDC wire ambiguity groups and
/// silicon overlap groups are expected to be small.
using DafMeasurementGroup = CkfTypes::BranchVector<SourceLink>;

/// Extension struct which holds delegates to customise the DAF behavior.
/// @tparam traj_t The trajectory type
template <typename traj_t>
struct DafFitterExtensions {
  /// Type alias for track state proxy from trajectory
  using TrackStateProxy = typename traj_t::TrackStateProxy;
  /// Type alias for const track state proxy from trajectory
  using ConstTrackStateProxy = typename traj_t::ConstTrackStateProxy;

  /// Reused unmodified from the Kalman fitter: calibrates a single source
  /// link into a track state.
  using Calibrator = typename KalmanFitterExtensions<traj_t>::Calibrator;
  /// Reused unmodified from the Kalman fitter: incorporates the effective
  /// measurement into the track parameters.
  using Updater = typename KalmanFitterExtensions<traj_t>::Updater;
  /// Reused unmodified from the Kalman fitter: back-propagates measurement
  /// information along the track.
  using Smoother = typename KalmanFitterExtensions<traj_t>::Smoother;

  /// Computes the chi2 of one already-calibrated candidate against the
  /// (shared) predicted state, before the candidates are combined.
  using ChiSquareCalculator = Delegate<double(ConstTrackStateProxy)>;

  /// Combines N calibrated candidates and their annealed weights into a
  /// single effective measurement written into the real track state.
  using WeightedCombiner =
      Delegate<void(const std::vector<ConstTrackStateProxy>&,
                    const std::vector<double>&, TrackStateProxy)>;

  /// Retrieves the associated surface from a source link
  SourceLinkSurfaceAccessor surfaceAccessor;

  /// Calibrates each candidate source link, called once per candidate
  Calibrator calibrator;

  /// Computes the pre-update chi2 of a calibrated candidate
  ChiSquareCalculator chiSquareCalculator;

  /// Combines calibrated, weighted candidates into the effective measurement
  WeightedCombiner weightedCombiner;

  /// Incorporates the effective measurement into the track parameters
  Updater updater;

  /// Back-propagates measurement information along the track
  Smoother smoother;

  /// Default constructor which connects the default void components
  DafFitterExtensions() {
    surfaceAccessor.connect<&detail::voidSurfaceAccessor>();
    calibrator.template connect<&detail::voidFitterCalibrator<traj_t>>();
    chiSquareCalculator
        .template connect<&detail::voidDafChiSquareCalculator<traj_t>>();
    weightedCombiner
        .template connect<&detail::voidDafWeightedCombiner<traj_t>>();
    updater.template connect<&detail::voidFitterUpdater<traj_t>>();
    smoother.template connect<&detail::voidFitterSmoother<traj_t>>();
  }
};

/// Per-surface persistent annealing state: the fixed candidate source links,
/// their current annealed weight, and their most recent chi2 -- carried by
/// the outer driver across forward+smooth passes since it cannot live in the
/// trajectory (only the effective combined measurement is written there).
struct DafCandidateState {
  /// The candidate source links, fixed for the whole fit
  DafMeasurementGroup candidates;
  /// Current annealed weight of each candidate, same order as candidates
  std::vector<double> weights;
  /// Most recent chi2 of each candidate, same order as candidates
  std::vector<double> chi2;
  /// Smoothed local0 of the track itself on this surface, from the pass that
  /// produced the current weights -- i.e. where the fit thinks it crossed
  /// this wire, in the same coordinate the candidates are expressed in.
  /// NaN until a weight update has run. Only meaningful alongside a
  /// DafCandidateTrace entry, where it makes the evolving track visible next
  /// to the evolving weights.
  double trackLoc0 = std::numeric_limits<double>::quiet_NaN();
};

/// Per-surface map of persistent annealing state, keyed the same way as the
/// input measurement groups.
using DafCandidateMap = std::unordered_map<const Surface*, DafCandidateState>;

/// One annealing pass's worth of candidate state, for tracing how the weights
/// evolve as the temperature falls.
///
/// The fitter overwrites its working DafCandidateMap in place on every pass,
/// so without capturing a copy per pass the annealing history exists only as
/// debug log text. That history is the whole content of "what the DAF did":
/// at a high temperature competing candidates share weight almost equally,
/// and only as the schedule cools does one of them win.
struct DafCandidateTraceEntry {
  /// Zero-based index of the forward+smooth pass these weights came out of
  std::size_t iteration = 0;
  /// The annealing temperature that pass ran at
  double temperature = 0.;
  /// Whether this is the result-producing pass (the final or salvaged one)
  bool isFinal = false;
  /// Copy of the per-surface state as it stood after that pass
  DafCandidateMap state;
};

/// The full annealing history, one entry per pass, in pass order.
using DafCandidateTrace = std::vector<DafCandidateTraceEntry>;

/// Combined options for the Deterministic Annealing Filter.
/// @tparam traj_t The trajectory type
template <typename traj_t>
struct DafFitterOptions {
  /// @param gctx The geometry context for this fit
  /// @param mctx The magnetic context for this fit
  /// @param cctx The calibration context for this fit
  /// @param extensions_ The DAF extensions
  /// @param pOptions The plain propagator options
  /// @param tSurface The target surface for the fit
  /// @param mScattering Whether to include multiple scattering
  /// @param eLoss Whether to include energy loss
  /// @param freeToBoundCorrection_ Correction for non-linearity effect during transform from free to bound
  /// @param annealingConfig_ The annealing temperature schedule and cutoff
  /// @param iterationsPerTemperature_ Number of forward+smooth passes run at each temperature before annealing
  /// @param maxIterations_ Hard cap on the total number of forward+smooth passes across the whole schedule
  DafFitterOptions(const GeometryContext& gctx,
                   const MagneticFieldContext& mctx,
                   std::reference_wrapper<const CalibrationContext> cctx,
                   DafFitterExtensions<traj_t> extensions_,
                   const PropagatorPlainOptions& pOptions,
                   const Surface* tSurface = nullptr, bool mScattering = true,
                   bool eLoss = true,
                   const FreeToBoundCorrection& freeToBoundCorrection_ =
                       FreeToBoundCorrection(false),
                   AnnealingUtility::Config annealingConfig_ =
                       AnnealingUtility::Config(),
                   std::size_t iterationsPerTemperature_ = 1,
                   std::size_t maxIterations_ = 100)
      : geoContext(gctx),
        magFieldContext(mctx),
        calibrationContext(cctx),
        extensions(extensions_),
        propagatorPlainOptions(pOptions),
        referenceSurface(tSurface),
        multipleScattering(mScattering),
        energyLoss(eLoss),
        freeToBoundCorrection(freeToBoundCorrection_),
        annealingConfig(std::move(annealingConfig_)),
        iterationsPerTemperature(iterationsPerTemperature_),
        maxIterations(maxIterations_) {}

  /// Context object for the geometry
  std::reference_wrapper<const GeometryContext> geoContext;
  /// Context object for the magnetic field
  std::reference_wrapper<const MagneticFieldContext> magFieldContext;
  /// Context object for the calibration
  std::reference_wrapper<const CalibrationContext> calibrationContext;

  /// Extensions for calibration, weighting and combination
  DafFitterExtensions<traj_t> extensions;

  /// The trivial propagator options
  PropagatorPlainOptions propagatorPlainOptions;

  /// The reference surface
  const Surface* referenceSurface = nullptr;

  /// Strategy to propagate to reference surface
  TrackExtrapolationStrategy referenceSurfaceStrategy =
      TrackExtrapolationStrategy::firstOrLast;

  /// Whether to consider multiple scattering
  bool multipleScattering = true;

  /// Whether to consider energy loss
  bool energyLoss = true;

  /// Whether to include non-linear correction during global to local
  /// transformation
  FreeToBoundCorrection freeToBoundCorrection;

  /// The annealing temperature schedule and cutoff
  AnnealingUtility::Config annealingConfig;

  /// Number of forward+smooth passes run at each temperature before
  /// annealing to the next one. Fixed count for M1; no weight-delta-based
  /// early stopping yet.
  std::size_t iterationsPerTemperature = 1;

  /// Hard cap on the total number of forward+smooth passes across the whole
  /// schedule, as a safety net independent of the temperature schedule
  /// length.
  std::size_t maxIterations = 100;

  /// Score the candidates against the UNBIASED smoothed state -- the smoothed
  /// state with this surface's own measurement removed again -- rather than the
  /// smoothed state itself.
  ///
  /// The weights decide which hypothesis a surface prefers, and scoring them
  /// against a state that already contains the blended measurement of those
  /// same hypotheses is self-confirming: where the blend sits near the wire,
  /// both mirrors score alike and the indecision sustains itself. GenFit scores
  /// against the unbiased state for this reason.
  ///
  /// The removal is an information subtraction, exact in principle but
  /// ill-conditioned when the measurement dominates the prior; the fitter falls
  /// back to the biased state whenever the result is not positive definite.
  bool useUnbiasedSmoothedState = false;

  /// Propagation steps allowed without forward progress before a pass is
  /// abandoned as stalled. The guard exists for oscillating intersection
  /// searches, but a low-momentum curler legitimately takes many steps between
  /// consecutive wires, so too small a value turns curlers into fit failures.
  std::size_t stallStepLimit = 2000;

  /// Factor the re-seed covariance is inflated by between annealing passes;
  /// 1 (the default) leaves the previous behaviour untouched.
  ///
  /// Each pass re-seeds from the previous pass's smoothed first-measurement
  /// state, which is the right thing for the PARAMETERS -- it re-linearises
  /// about the current best trajectory. Carrying the smoothed COVARIANCE over
  /// as well is not: that covariance already contains every measurement on the
  /// track, and the pass then filters those same measurements again, so their
  /// information is counted twice, and once more on every further pass. The
  /// predicted covariances shrink accordingly, chi2 values inflate, and the
  /// annealing starts reading good hits as outliers.
  ///
  /// GenFit avoids this by blowing the covariance up between iterations
  /// (AbsKalmanFitter's blowUpFactor, default 1e3). Scaling preserves the
  /// correlation coefficients while removing the false confidence; GenFit
  /// additionally resets the off-diagonals, which is not done here.
  double reseedCovarianceScaling = 1.;

  /// Linearise about a reference trajectory instead of about the running
  /// filtered state -- GenFit's KalmanFitterRefTrack rather than its plain
  /// KalmanFitter, which is what GenFit's own DAF runs.
  ///
  /// With this set, each annealing pass is split in two. A reference pass
  /// propagates without ever feeding the filtered state back into the stepper,
  /// recording per surface the reference parameters, the transport Jacobian F,
  /// the process noise N and the delta state c, and calibrating every candidate
  /// against that reference. A filter pass then applies the stored affine map
  /// p = F*p + c, C = F*C*F^T + N and updates -- without propagating at all.
  ///
  /// The consequence is that every hit in a pass shares one linearisation
  /// point, and that point is a full-track-length trajectory rather than a
  /// partially-informed running estimate. It matters most where the measurement
  /// itself depends on the expansion point, i.e. drift chambers, whose x-t
  /// relation and left/right hypotheses are evaluated at the incidence angle of
  /// whatever state the calibrator is handed.
  ///
  /// The reference is updated between passes by re-seeding from the previous
  /// pass's smoothed first-measurement state, i.e. every pass re-linearises --
  /// GenFit's setRefitAll, which is what its DAF uses. GenFit's selective
  /// deltaChi2Ref update is therefore deliberately not implemented: the DAF
  /// never uses it.
  ///
  /// One deviation from GenFit remains. GenFit rebuilds its reference
  /// piecewise, extrapolating each segment from the previous point's SMOOTHED
  /// state, so the reference hugs the smoothed trajectory over the whole track.
  /// Here the reference is one propagation from the smoothed first-measurement
  /// state, so it can drift from the smoothed curve over a long track. Matching
  /// GenFit exactly would need one short propagation per segment.
  bool referenceLinearization = false;

  /// Score the candidates for the weight update using the measurements the
  /// REFERENCE pass calibrated, rather than re-calibrating them against the
  /// smoothed state. No effect unless referenceLinearization is set.
  ///
  /// This is what makes reference linearisation cover the whole pass rather
  /// than only its filter. GenFit calibrates each surface's measurements once,
  /// in KalmanFitterRefTrack::prepareTrack, against that surface's reference
  /// state (@c constructMeasurementsOnPlane(*referenceState)), and DAF's
  /// processWeights then scores those same measurement objects against the
  /// smoothed state. One linearisation point per pass, used by filter and
  /// weights alike.
  ///
  /// ACTS's DAF instead re-calibrates every candidate against the smoothed
  /// state before scoring it. For a silicon hit that is a no-op -- the
  /// calibration does not depend on the expansion point -- but a drift-chamber
  /// hit's x-t relation is evaluated at the incidence angle of whatever state
  /// the calibrator is handed, so the drift length entering the weight chi2 is
  /// then a different number from the one the filter fitted. That inconsistency
  /// is exactly what reference linearisation exists to remove, which is why
  /// this defaults to on: with it, the recorded candidates are reused and only
  /// their expansion point is swapped for the smoothed state, matching GenFit.
  ///
  /// Turn it off to isolate the filter half of the change; the fit then
  /// linearises its filter about the reference but keeps scoring weights
  /// against smoothed-state calibrations.
  bool referenceWeightCalibration = true;

  /// Factor applied to the seed covariance the reference filter starts each
  /// pass from. Leave it at 1 unless you know what you are doing.
  ///
  /// This is NOT GenFit's blowUpFactor, and must not be set to 1e3 to imitate
  /// it. GenFit can inflate its covariance by 1e3 because its state is
  /// (q/p, u', v', u, v) -- every component unbounded, so scaling a slope
  /// variance by 1000 is meaningless but harmless. An ACTS bound covariance
  /// contains var(phi) and var(theta), which are BOUNDED quantities: inflating
  /// var(theta) from 1e-2 to 10 rad^2 asserts a direction prior spanning
  /// several revolutions. The first update then throws the direction anywhere,
  /// normalizePhiTheta folds the result back into range (flipping phi by pi
  /// whenever theta wraps past 0 or pi), and the next pass's reference
  /// propagation starts outside the geometry -- "no start volume could be
  /// resolved", plus failed field lookups. Observed, not hypothetical.
  ///
  /// The blow-up is unnecessary here anyway. GenFit needs it because it starts
  /// from a covariance that is not valid where it is applied;
  /// detail::dafReferenceFilter instead TRANSPORTS the seed covariance onto the
  /// first surface (J*C*J^T + N), which is exactly the covariance the
  /// running-state filter has at that point -- correctly placed, no double
  /// counting, and no inflation required.
  double referenceStartCovarianceScaling = 1.;

  /// Drift-circle initial weighting on straw surfaces, as a multiple of the
  /// straw's radial cell spacing; 0 (the default) keeps the uniform 1/N start.
  /// Mirrors GenFit's CDCRecoHit, which starts a mirror pair at
  /// w = 0.5 * max(0, 1 - rbar/rMax)^2 so that the first pass distrusts the
  /// blended-on-the-wire measurement in proportion to the drift radius.
  /// GenFit uses rMax = 1.5 * cell spacing, so 3 (this factor multiplies HALF
  /// the cell spacing, recovered from the straw bounds) matches it exactly. See
  /// DafActor::strawInitialWeightRadiusFactor for the full reasoning.
  double strawInitialWeightRadiusFactor = 0.;

  /// Optional hook to rebuild the DirectNavigator surface sequence from each
  /// pass's re-seeded parameters, instead of reusing the one sequence built
  /// from the original seed for every pass.
  ///
  /// The re-seeding above updates the track PARAMETERS between passes, but the
  /// sequence -- which surfaces, in what order, at what path lengths -- stays
  /// frozen at the initial seed's geometry, because it is prepared once by the
  /// caller and handed to fit(). Every pass therefore re-linearises the fit
  /// against an extrapolation geometry derived from a seed it has already
  /// improved on. GenFit's KalmanFitterRefTrack rebuilds its reference track on
  /// every iteration (setRefitAll), so its geometry follows the current
  /// estimate; this hook is the ACTS equivalent.
  ///
  /// Unset (the default) reproduces the previous behaviour exactly, including
  /// for fits that do not use a DirectNavigator at all.
  ///
  /// The returned sequence MUST still contain every measurement-group surface.
  /// DafActor finishes a pass when
  /// measurementStates == inputMeasurementGroups.size(), so a hit surface
  /// missing from the sequence leaves that condition permanently unsatisfied
  /// and the propagation runs on to the path limit -- turning a re-linearised
  /// pass into a lost track. The fitter verifies this and keeps the previous
  /// pass's sequence when the rebuilt one would drop a hit surface, so a
  /// rebuilder that also applies a reachability cut degrades to the old
  /// behaviour rather than to a failure.
  std::function<std::vector<const Surface*>(const BoundTrackParameters&)>
      rebuildSequence;

  /// Optional out-parameter: when non-null, the fitter copies the final
  /// per-surface candidate state (weights/chi2 as consumed by the pass that
  /// produced the returned track, including a salvaged one) into it. The
  /// weights are otherwise internal to the annealing loop and discarded.
  /// Not a constructor parameter -- set it after construction; all existing
  /// construction sites stay valid.
  DafCandidateMap* finalCandidateState = nullptr;

  /// Optional out-parameter: when non-null, the fitter appends one entry per
  /// annealing pass, recording the weights that pass produced and the
  /// temperature it ran at. Same "set it after construction" pattern as
  /// finalCandidateState, and likewise free when left null -- copying the
  /// candidate map per pass is only worth it when something is going to plot
  /// it.
  DafCandidateTrace* candidateTrace = nullptr;
};

/// @}

}  // namespace Acts

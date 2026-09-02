// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"

#include <array>

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

  /// Log of the Gaussian prefactor 1/sqrt((2*pi)^d * det S) that belongs with
  /// the same candidate's chi2, for GenFit-equivalent weights. Must be
  /// connected with the SAME DafChiSquareDenominator as the chi2 calculator.
  using LogNormalization = Delegate<double(ConstTrackStateProxy)>;

  /// Combines N calibrated candidates and their annealed weights into a
  /// single effective measurement written into the real track state.
  using WeightedCombiner =
      Delegate<void(const std::vector<ConstTrackStateProxy>&,
                    const std::vector<double>&, TrackStateProxy)>;

  /// Retrieves the associated surface from a source link
  SourceLinkSurfaceAccessor surfaceAccessor;

  /// Calibrates each candidate source link, called once per candidate
  Calibrator calibrator;

  /// Computes the pre-update chi2 of a calibrated candidate.
  ///
  /// Which covariance that chi2 divides by is NOT a field on this struct -- it
  /// is fixed by which calculator the caller connects here, see
  /// detail::DafChiSquareDenominator. Deliberately: an options field would have
  /// to be read by the delegate, which only receives a track state, so it could
  /// only ever be silently ignored.
  ChiSquareCalculator chiSquareCalculator;

  /// Per-candidate Gaussian normalisation, only consulted when
  /// DafFitterOptions::useCandidateNormalization is set.
  LogNormalization logNormalization;

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
    logNormalization
        .template connect<&detail::voidDafLogNormalization<traj_t>>();
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

  /// Where this pass linearised the fit on this surface, and with what
  /// transport. Filled only while a DafCandidateTrace is being recorded --
  /// see DafFitterOptions::candidateTrace -- because it is only ever read
  /// per-pass, and a fit that is not tracing has nothing to compare against.
  /// @c hasLinearization says whether the rest of this block means anything.
  ///
  /// This is the raw material for asking how much a plain (running-state)
  /// EKF's expansion point moves between annealing passes, which is exactly
  /// what a reference track would hold still. A pass whose predicted state,
  /// transport Jacobian and measurement frame barely move from the previous
  /// one had nothing to gain from a fixed reference; one where they move by
  /// more than the measurement resolution did.
  bool hasLinearization = false;
  /// Predicted (pre-update) bound parameters on this surface, i.e. the
  /// expansion point the filter actually linearised the update about.
  BoundVector predictedParameters = BoundVector::Zero();
  /// Smoothed (post-update, whole-track) bound parameters on this surface.
  /// The last pass's value of this is the best available stand-in for the
  /// converged trajectory that a reference track would be built from.
  BoundVector smoothedParameters = BoundVector::Zero();
  /// Bound-to-bound transport Jacobian from the previous track state to this
  /// one -- F_{m-1} in the linearised filter. Identity on the first state of
  /// a pass, where there is no previous state to transport from.
  BoundMatrix jacobian = BoundMatrix::Identity();
  /// Global position of the smoothed crossing point on this surface.
  Vector3 globalPosition = Vector3::Zero();
  /// First column of the surface's measurement (reference) frame at that
  /// crossing -- the direction loc0 is measured along, i.e. the direction the
  /// projector H picks out. For a straw this is the wire-to-track direction,
  /// which rotates with the track's momentum direction and so is *not* fixed
  /// by the geometry alone; that rotation is the third linearisation
  /// instability a reference track removes.
  Vector3 measurementAxis = Vector3::Zero();
};

/// Per-surface map of persistent annealing state, keyed the same way as the
/// input measurement groups.
using DafCandidateMap = std::unordered_map<const Surface*, DafCandidateState>;

/// What one weight-update pass observed, for the convergence tests.
struct DafPassSummary {
  /// Largest absolute weight change over every candidate on every surface --
  /// GenFit's maxAbsChange in DAF::calcWeights, its primary convergence
  /// measure.
  double maxAbsWeightChange = 0.;
  /// Weight-averaged chi2 summed over every surface, evaluated against the
  /// SMOOTHED state. The closest thing ACTS has to GenFit's backward chi2:
  /// both score the measurements against a state that has seen the whole
  /// track, so both are biased low against an independent estimate, but they
  /// are not the same accumulation and the p-values built from them are not
  /// numerically identical. See DafFitterOptions::pValueConvergenceTolerance.
  double weightedChi2 = 0.;
  /// Measurement dimensions summed over the surfaces, minus the five track
  /// parameters. Zero or negative means the track is not over-constrained and
  /// no p-value can be formed.
  double ndf = 0.;
};

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

  /// Weight every candidate by its own Gaussian prefactor
  /// norm = 1/sqrt((2*pi)^d * det S), as GenFit does (DAF.cc:313).
  ///
  /// ON by default (changed 2026-08-28), because it is what GenFit does and it
  /// costs nothing where it does not apply: it is an exact NO-OP wherever the
  /// competing candidates on a surface share one S, which is the case for any
  /// detector whose candidates are the same measurement type with the same
  /// covariance. There nu divides out of numerator and denominator identically.
  ///
  /// It matters for CDC drift circles, where the left and right mirror
  /// hypotheses carry separately evaluated resolutions. Verified in the ACTS
  /// input rather than assumed: of 292 straws in one event, var_local0 differs
  /// between the two mirrors on 290, by a median 3.7% and up to 144% -- so the
  /// per-side resolutions basf2 derives from (z, alpha, theta) are already
  /// present in the measurement file. Over 64948 pairs the resulting weight
  /// tilt sqrt(S_b/S_a) is a median 1.5%, exceeds 5% on 20% of pairs and 25%
  /// on 2.7%.
  ///
  /// GenFit's own comment says why it cannot be skipped: "the following assumes
  /// that in the competing hits could have different V" (DAF.cc:313, where the
  /// same norm scales the cut-off term too).
  ///
  /// Requires DafFitterExtensions::logNormalization to be connected with the
  /// same denominator as the chi2 calculator; with the void one it degrades to
  /// the norm-free weights rather than failing.
  bool useCandidateNormalization = true;

  /// Stop annealing early once no candidate weight moves by more than this
  /// between consecutive passes; 0 (the default) runs the whole schedule.
  ///
  /// GenFit's deltaWeight_, 0.001 in basf2's payload: DAF::calcWeights returns
  /// whether every weight changed by less than it, and
  /// DAF::processTrackWithRep then does ONE further pass with the converged
  /// weights and stops. A track whose weights settle at pass 5 costs 6 passes
  /// there and 9 here.
  ///
  /// Not purely a CPU knob: every pass re-seeds the next from its smoothed
  /// first-measurement state, so passes beyond convergence keep moving the
  /// linearisation point without any weight change to justify it.
  double weightConvergenceTolerance = 0.;

  /// Passes that must run before either convergence test may stop the fit.
  /// GenFit's minIterations_, 5 in basf2's payload. Without it a fit whose
  /// weights happen not to move on pass 1 -- every surface with a single
  /// candidate, say -- would stop while still hot.
  std::size_t minIterations = 1;

  /// GenFit's SECONDARY convergence test: accept a pass whose weights are still
  /// moving, provided the fit p-value is valid and has settled. 0 (the default)
  /// disables it.
  ///
  /// The test is a fallback, evaluated only when the weight test above has
  /// failed, and is four conditions ANDed (DAF.cc:153-155):
  ///
  ///   pass >= pValueMinIterations && p > pValueMinimum
  ///     && lastP >= pValueMinimum && |lastP - p| < this
  ///
  /// basf2's payload sets deltaPValue = 1, which with p in [0, 1] makes the
  /// STABILITY half vacuous and leaves the VALIDITY half doing the work: `p >
  /// pValueMinimum` is a strict inequality, so a fit whose p-value underflows
  /// to exactly zero is excluded and keeps annealing. Reproducing GenFit
  /// therefore means setting this to 1, not to something small -- a small value
  /// is a genuinely different, stricter test.
  ///
  /// CAVEAT ON THE P-VALUE ITSELF. GenFit uses the BACKWARD pass's p-value.
  /// ACTS has no backward chi2; the closest available quantity, and what this
  /// uses, is DafPassSummary::weightedChi2, the weight-averaged chi2 against
  /// the SMOOTHED state. Both are scored against a state that has seen the
  /// whole track, so both are biased low, but they are not the same
  /// accumulation and the two p-values are not numerically identical.
  double pValueConvergenceTolerance = 0.;

  /// The p-value a pass must strictly exceed for the test above to accept it.
  /// GenFit's minPval_, 0 in basf2's payload -- which is not a no-op, because
  /// the comparison is strict.
  double pValueMinimum = 0.;

  /// Passes that must run before the p-value test may fire. GenFit's
  /// minIterForPval_, 5 in basf2's payload. Independent of minIterations.
  std::size_t pValueMinIterations = 5;

  /// Reset the off-diagonals of the re-seed covariance, as GenFit's
  /// MeasuredStateOnPlane::blowUpCov does (resetOffDiagonals defaults true
  /// there). Keeping them preserves correlations that came from the very
  /// measurements the next pass is about to re-filter, which is part of what
  /// the blow-up exists to discard. Measured to contribute nothing on its own.
  bool reseedResetOffDiagonals = false;

  /// Per-variance ceiling applied after scaling, GenFit's blowUpMaxVal (1e6
  /// there). 0 disables it.
  ///
  /// This is what makes a large scaling safe, and why GenFit can use 1000 where
  /// ACTS cannot. GenFit's plane state is (q/p, u', v', u, v) -- every
  /// component unbounded, so a variance of 1e6 simply means "no information".
  /// An ACTS bound state carries phi and theta, which are BOUNDED: once
  /// sigma_theta approaches pi the prior is already flat, and past that it is
  /// nonsense. At scaling 1000 sigma_theta reaches order 10 rad, the first
  /// update throws the direction anywhere, normalizePhiTheta folds theta back
  /// into [0, pi] and flips phi by pi on the way, and the pass restarts from a
  /// state pointing somewhere arbitrary. Observed as a large efficiency loss.
  ///
  /// Ceilings are given per parameter so the unbounded components can take
  /// GenFit's full blow-up while the angles stay inside the branch cut. See
  /// reseedVarianceCeilings.
  double reseedVarianceCeiling = 0.;

  /// Per-parameter ceilings, indexed by BoundIndices, overriding the scalar
  /// reseedVarianceCeiling where non-zero. The point of the split: loc0, loc1
  /// and q/p are unbounded and can absorb the full 1000x, while eBoundPhi and
  /// eBoundTheta must stay well inside their periodic range -- (0.1 rad)^2 is
  /// already an almost flat direction prior and is nowhere near the wrap.
  std::array<double, eBoundSize> reseedVarianceCeilings{};

  /// Smooth by running a second Kalman filter BACKWARD and combining it with
  /// the forward one, instead of the RTS gain-matrix smoother. GenFit's
  /// arrangement (KalmanFitterRefTrack::processTrackWithRep: forward filter,
  /// blow up the last point, backward filter) with GenFit's combination
  /// (KalmanFitterInfo::getFittedState(biased=true)):
  ///
  ///   C_s = (C_f^-1 + C_b^-1)^-1
  ///   x_s = C_s (C_f^-1 x_f + C_b^-1 x_b)
  ///
  /// where x_f is the forward FILTERED state and x_b the backward PREDICTED
  /// one, so each measurement enters exactly once.
  ///
  /// Off by default. For a linear-Gaussian system this is algebraically the
  /// same smoothed estimate the RTS smoother produces, so it is expected to
  /// change nothing except through non-linearity -- and it is the less
  /// numerically stable of the two, since it inverts covariances to combine.
  /// Any failure (no reverse pass, a surface with no backward partner, a
  /// non-invertible sum) falls back to the RTS smoother for that track.
  bool useTwoFilterSmoothing = false;

  /// Covariance blow-up applied to the backward filter's starting state, at
  /// the last measurement. GenFit's blowUpFactor, 1e3. The ceilings above are
  /// applied with it -- an ACTS bound state has periodic phi and theta, so an
  /// unclipped blow-up here wraps them exactly as it does at the re-seed.
  double twoFilterBlowUpFactor = 1000.;

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

  /// Build the reference trajectory SEGMENT BY SEGMENT -- one short
  /// propagation from each sequence surface to the next -- instead of
  /// capturing one continuous propagation from the seed. No effect unless
  /// referenceLinearization is set.
  ///
  /// GenFit's KalmanFitterRefTrack::prepareTrack works this way
  /// (KalmanFitterRefTrack.cc:477-498: extrapolateToPlane from the previous
  /// reference state to the next plane, then getForwardJacobianAndNoise per
  /// segment). ACTS's single propagation is cheaper but drifts from the curve
  /// the fit actually converged to over a long track, which is the reference
  /// linearisation's whole reason to exist.
  ///
  /// On its own this should be a NO-OP: the segment boundaries fall exactly
  /// where the continuous recording already divides the noise (surface m's
  /// pre-update material into N_m, its post-update material into N_m+1), so
  /// the two should produce the same trajectory. That equality is the
  /// invariant referencePiecewiseSelfCheck verifies, and it is what catches
  /// the failure mode this split can have: a propagation STARTING on surface m
  /// skipping m's post-update material, losing noise at every boundary.
  bool referencePiecewise = false;

  /// Re-linearise each surface's reference around the PREVIOUS pass's smoothed
  /// state at that surface, rather than around the transported seed. Requires
  /// referencePiecewise.
  ///
  /// This is the other half of GenFit's arrangement: with setRefitAll() every
  /// DAF iteration rebuilds the reference, and a point that has a smoothed
  /// state re-linearises around it (KalmanFitterRefTrack.cc:531/588 --
  /// constructPlane(*smoothedState), then extrapolate that state to the plane).
  /// ACTS re-seeds only from the smoothed state at the FIRST measurement and
  /// propagates onward, so surfaces far from it are linearised about a curve
  /// the fit has already moved away from.
  ///
  /// The transport F_m and noise N_m still come from propagating the previous
  /// reference to surface m; only the linearisation POINT at m is replaced.
  /// The affine constant c_m = p_ref,m - F_m p_ref,m-1 absorbs the difference,
  /// which is exactly what it is for.
  bool referenceAnchorSmoothed = false;

  /// Record both the piecewise and the continuous reference on the first pass
  /// and report where they differ. Diagnostic only, and expensive -- it doubles
  /// the recording work.
  bool referencePiecewiseSelfCheck = false;

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
  ///
  /// The reference path has NO blow-up options of its own: it uses the same
  /// reseedCovarianceScaling / reseedResetOffDiagonals / reseedVarianceCeiling
  /// / reseedVarianceCeilings the running-state path does, applied to whichever
  /// prior a pass filters against (the first pass's seed covariance, and the
  /// transported prior cached for passes 2..N).
  ///
  /// Sharing them is what makes a comparison between the two paths fair.
  /// Running one at 10 and the other at 1000 is not an A/B of reference
  /// linearisation, it is an A/B of the prior -- and a prior that is too tight
  /// or too loose shows up exactly where there are too few hits to overwhelm
  /// it, i.e. on short tracks, which is where the 2026-08-22 comparison lost
  /// the most. Before 2026-09-01 the two families were separate and defaulted
  /// differently (1000/reset-on against 10/reset-off), so every reference-vs-
  /// running comparison before then carried that difference too.

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

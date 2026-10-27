// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFitting/DafFitterFunction.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/TrackFitting/DeterministicAnnealingFitter.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/MbfSmoother.hpp"
#include "Acts/TrackFitting/detail/DafChiSquareCalculator.hpp"
#include "Acts/TrackFitting/detail/DafWeightedCombiner.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/TrackFitting/DirectNavigationSequence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <memory>
#include <utility>
#include <vector>

namespace {

using Stepper = Acts::SympyStepper;
using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
using Fitter =
    Acts::DeterministicAnnealingFitter<Propagator, Acts::VectorMultiTrajectory>;
using DirectPropagator = Acts::Propagator<Stepper, Acts::DirectNavigator>;
using DirectFitter =
    Acts::DeterministicAnnealingFitter<DirectPropagator,
                                       Acts::VectorMultiTrajectory>;

using TrackContainer =
    Acts::TrackContainer<Acts::VectorTrackContainer,
                         Acts::VectorMultiTrajectory, std::shared_ptr>;


namespace {
/// Tuning knobs for the direct-navigation fit, read once from the environment
/// so a single build can be A/B tested without a rebuild:
///   ACTS_MATERIAL_VOLUMES="19-74"  -- restrict the material surfaces offered
///        to the sequence to these geometry volume ids (comma separated ids or
///        lo-hi ranges). In the Belle II geometry 14-18 are the 172 SVD sensor
///        modules, 19-74 the 56 CDC layer cylinders, 1-13 beam pipe/PXD. A
///        CDC-only fit does not need the SVD sensors, and every extra surface
///        is one more the DirectNavigator has to walk. Empty => no restriction.
///   ACTS_MAX_STEPS=N               -- override the propagator step budget
///        (Acts default 15000). Empty => leave as configured.
std::vector<std::pair<unsigned, unsigned>> materialVolumeRanges() {
  static const std::vector<std::pair<unsigned, unsigned>> ranges = [] {
    std::vector<std::pair<unsigned, unsigned>> out;
    const char* env = std::getenv("ACTS_MATERIAL_VOLUMES");
    if (env == nullptr || *env == 0) {
      return out;
    }
    std::string spec(env);
    std::size_t pos = 0;
    while (pos < spec.size()) {
      std::size_t comma = spec.find(',', pos);
      std::string tok = spec.substr(pos, comma == std::string::npos
                                             ? std::string::npos
                                             : comma - pos);
      std::size_t dash = tok.find('-');
      if (dash == std::string::npos) {
        const unsigned v = std::strtoul(tok.c_str(), nullptr, 10);
        out.emplace_back(v, v);
      } else {
        out.emplace_back(std::strtoul(tok.substr(0, dash).c_str(), nullptr, 10),
                         std::strtoul(tok.substr(dash + 1).c_str(), nullptr, 10));
      }
      if (comma == std::string::npos) {
        break;
      }
      pos = comma + 1;
    }
    return out;
  }();
  return ranges;
}

bool materialVolumeAccepted(unsigned volume) {
  const auto& ranges = materialVolumeRanges();
  if (ranges.empty()) {
    return true;
  }
  for (const auto& [lo, hi] : ranges) {
    if (volume >= lo && volume <= hi) {
      return true;
    }
  }
  return false;
}

/// Steps without forward progress before a pass is abandoned (0 = keep the
/// DafFitterOptions default). Raising it is what a multi-turn curler needs:
/// it takes many steps between consecutive wires, which the guard cannot
/// distinguish from a non-converging target search.
unsigned stallStepsOverride() {
  static const unsigned value = [] {
    const char* env = std::getenv("ACTS_DAF_STALL_STEPS");
    return env == nullptr ? 0u
                          : static_cast<unsigned>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// Multiplies the path limit derived from the sequence span. A curler's arc
/// length grows as its momentum falls, so a limit sized for a straight track
/// truncates it mid-trajectory and it ends with too few measurements.
/// Weight each DAF candidate by its own Gaussian prefactor
/// 1/sqrt((2*pi)^d det S), as GenFit does. ON by default; set
/// ACTS_DAF_CAND_NORM=0 to fall back to the norm-free weights.
///
/// A no-op unless the competing candidates on a surface have different
/// covariances -- which for CDC drift circles they do: the left and right
/// mirrors carry separately evaluated drift resolutions, already present in the
/// measurement file.
bool useCandidateNormalization() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DAF_CAND_NORM");
    return env == nullptr || std::atoi(env) != 0;
  }();
  return value;
}

double pathLimitScale() {
  static const double value = [] {
    const char* env = std::getenv("ACTS_DAF_PATH_SCALE");
    return env == nullptr ? 1.0 : std::strtod(env, nullptr);
  }();
  return value;
}

/// Restore Acts' loop protection on the direct-navigation fit
/// (ACTS_LOOP_PROTECTION=1), which this branch otherwise switches off. The
/// KF reads the same variable -- this is a propagator-level concern, not a DAF
/// one, and KalmanFitterFunction.cpp carries the identical block.
///
/// Loop protection caps the propagation at `loopFraction * 2*pi*p/B` -- half a
/// helix turn with the Acts defaults (PropagatorOptions.hpp: loopProtection =
/// true, loopFraction = 0.5). It is the right guard for the geometric Navigator,
/// which has no idea how far it should go, but here the prepared sequence
/// already bounds the propagation and pathLimit below is sized from its span.
/// The two do not compose: `Propagator::initialize` lowers the propagator's
/// aborter to the loop limit, and `DafActor::act` sets its OWN result-side
/// aborter to it outright (releaseLimit = true), so the computed pathLimit is
/// overridden and no pass can span more than half a turn. When that limit
/// fires DafActor sets `finished = true` -- a clean finish, not an error -- so
/// the remaining sequence surfaces are never visited, keep the 1/N seed weights
/// the fitter gave them up front, and are not even counted as holes.
///
/// Measured 2026-08-26 on acts_daf_blowup_noreset, 400 events: 64.4% of the
/// hits that reach the export unprocessed lie past the half-turn limit against
/// 0.81% of processed hits, the limit undercuts the pathLimit computed here on
/// 7.8% of tracks, and the unprocessed rate runs 13-27% below 300 MeV against
/// 0.1-0.3% above 500 MeV. Note this also makes ACTS_DAF_PATH_SCALE testable:
/// while loop protection clamps below it, that knob cannot do anything.
///
/// Switching it off does not remove path limiting. `Propagator::makeState`
/// appends a PathLimitReached aborter carrying `options.pathLimit`
/// (Propagator.ipp:255), so the limit computed from the sequence span governs;
/// `setupLoopProtection` early-returns on this flag, leaving DafActor's own
/// aborter inert at its max() default.
bool loopProtection() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_LOOP_PROTECTION");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}

/// Score DAF candidate weights against the unbiased smoothed state.
bool useUnbiasedSmoothedState() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DAF_UNBIASED");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}

/// Rebuild the DirectNavigator sequence from each pass's re-seeded parameters
/// (ACTS_DAF_REBUILD_SEQ=1) instead of reusing the one built from the original
/// seed. The re-seeding already re-linearises the PARAMETERS between passes;
/// this re-linearises the navigation GEOMETRY with them, which is what GenFit's
/// KalmanFitterRefTrack does via setRefitAll. Costs one sequence preparation
/// per pass -- a sort plus a material-surface sweep -- so it is not free.
bool rebuildSequencePerPass() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DAF_REBUILD_SEQ");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}

/// Re-order the DirectNavigator sequence between passes from the trajectory
/// the previous pass actually propagated (ACTS_DAF_RESORT=1): reached surfaces
/// at their reached path length, missed ones placed by the fitted track, none
/// dropped. See Acts::DafFitterOptions::resortSequence and
/// resortDirectNavigationSequence. Takes precedence over ACTS_DAF_REBUILD_SEQ.
bool resortSequencePerPass() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DAF_RESORT");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}

unsigned maxStepsOverride() {
  static const unsigned value = [] {
    const char* env = std::getenv("ACTS_MAX_STEPS");
    return env == nullptr ? 0u
                          : static_cast<unsigned>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// Energy loss on the extrapolation from the first hit to the perigee,
/// switched separately from the fit's: ACTS_O2_ENERGY_LOSS=0 turns it off, =1
/// on, and unset follows the fit's own energyLoss flag as before. The fit and
/// this extrapolation otherwise share one switch, so an energyLoss=False run
/// cannot tell which of the two leaves the ~0.7 MeV momentum offset
/// (FIT_CHANGES Q12).
bool extrapolationEnergyLoss(bool fitEnergyLoss) {
  static const int value = [] {
    const char* env = std::getenv("ACTS_O2_ENERGY_LOSS");
    return env == nullptr ? -1 : (std::atoi(env) != 0 ? 1 : 0);
  }();
  return value < 0 ? fitEnergyLoss : value == 1;
}

/// Directory to dump the material the perigee extrapolation applies, per track
/// and per surface, or nullptr (the default) for no dump. ACTS_O2_MATERIAL_DUMP
/// names it; two files are written there, o2-tracks.csv and o2-material.csv.
/// For locating the ~0.7 MeV the extrapolation fails to add back (FIT_CHANGES
/// Q12): the per-surface list shows which surface is missing or under-applied.
const char* o2MaterialDumpDir() {
  static const char* value = [] {
    const char* env = std::getenv("ACTS_O2_MATERIAL_DUMP");
    return (env != nullptr && *env != '\0') ? env : nullptr;
  }();
  return value;
}

/// Append one extrapolated track to the dump. `seq` counts dumped tracks over
/// the whole run; a row is matched to its fitted track through the perigee
/// parameters (end_*), which are exactly what track_parameters.csv holds.
///
/// o2-material.csv has one row per material interaction, in the order the
/// extrapolation met them. thickness_x0 is the slab as applied: already scaled
/// by path_correction and by the start/target update-mode factor, so the plain
/// surface material is thickness_x0 / path_correction. dp_mev is the momentum
/// before minus after; an inward extrapolation that restores energy gives
/// negative values.
void writeO2Material(const Acts::BoundTrackParameters& start,
                     const Acts::BoundVector& end,
                     const Acts::RecordedMaterial& material, double distance,
                     const Acts::GeometryContext& gctx) {
  using namespace Acts::UnitLiterals;
  static std::mutex mutex;
  static std::ofstream tracksCsv;
  static std::ofstream materialCsv;
  static std::size_t seq = 0;
  std::lock_guard<std::mutex> lock(mutex);
  if (!tracksCsv.is_open()) {
    const std::string dir = o2MaterialDumpDir();
    tracksCsv.open(dir + "/o2-tracks.csv");
    materialCsv.open(dir + "/o2-material.csv");
    tracksCsv << "seq,start_x,start_y,start_z,start_r,start_p,distance,"
                 "end_d0,end_z0,end_phi,end_theta,end_qop,n_interactions,"
                 "x0_total,dp_total_mev\n";
    materialCsv << "seq,geometry_id,volume,boundary,layer,approach,sensitive,"
                   "x,y,z,r,path_correction,thickness_x0,dp_mev\n";
    tracksCsv << std::setprecision(12);
    materialCsv << std::setprecision(8);
  }

  double dpTotal = 0.;
  for (const auto& mi : material.materialInteractions) {
    dpTotal += mi.deltaP;
    const Acts::GeometryIdentifier id = (mi.surface != nullptr)
                                            ? mi.surface->geometryId()
                                            : mi.intersectionID;
    materialCsv << seq << ',' << id.value() << ',' << id.volume() << ','
                << id.boundary() << ',' << id.layer() << ',' << id.approach()
                << ',' << id.sensitive() << ',' << mi.position.x() << ','
                << mi.position.y() << ',' << mi.position.z() << ','
                << std::hypot(mi.position.x(), mi.position.y()) << ','
                << mi.pathCorrection << ','
                << mi.materialSlab.thicknessInX0() << ','
                << mi.deltaP / 1_MeV << '\n';
  }
  const Acts::Vector3 pos = start.position(gctx);
  tracksCsv << seq << ',' << pos.x() << ',' << pos.y() << ',' << pos.z() << ','
            << std::hypot(pos.x(), pos.y()) << ','
            << start.absoluteMomentum() << ',' << distance << ','
            << end[Acts::eBoundLoc0] << ',' << end[Acts::eBoundLoc1] << ','
            << end[Acts::eBoundPhi] << ',' << end[Acts::eBoundTheta] << ','
            << end[Acts::eBoundQOverP] << ','
            << material.materialInteractions.size() << ','
            << material.materialInX0 << ',' << dpTotal / 1_MeV << '\n';
  ++seq;
}

/// Acts::extrapolateTrackToReferenceSurface, step for step, but keeping the
/// propagation result so the recorded material can be dumped: the library
/// helper discards it.
template <typename track_proxy_t, typename propagator_t, typename options_t>
Acts::Result<void> extrapolateAndDumpMaterial(
    track_proxy_t track, const Acts::Surface& referenceSurface,
    const propagator_t& propagator, options_t options,
    Acts::TrackExtrapolationStrategy strategy,
    const Acts::GeometryContext& gctx) {
  auto findResult = Acts::findTrackStateForExtrapolation(
      gctx, track, referenceSurface, strategy);
  if (!findResult.ok()) {
    return findResult.error();
  }
  auto& [trackState, distance] = *findResult;
  options.direction = Acts::Direction::fromScalarZeroAsPositive(distance);
  const Acts::BoundTrackParameters start =
      track.createParametersFromState(trackState);

  auto propagated =
      propagator.template propagate<options_t, Acts::ForcedSurfaceReached>(
          start, referenceSurface, options);
  if (!propagated.ok()) {
    return propagated.error();
  }
  track.setReferenceSurface(referenceSurface.getSharedPtr());
  track.parameters() = propagated->endParameters.value().parameters();
  track.covariance() =
      propagated->endParameters.value().covariance().value();

  writeO2Material(
      start, track.parameters(),
      propagated->template get<Acts::MaterialInteractor::result_type>(),
      distance, gctx);
  return Acts::Result<void>::success();
}
}  // namespace

using namespace ActsExamples;

struct DafFitterFunctionImpl final : public DafFitterFunction {
  Fitter fitter;
  DirectFitter directFitter;
  // Plain geometric-Navigator propagator used only for the final
  // extrapolation to the reference surface after a direct-navigation fit
  // (the DirectNavigator cannot target a surface outside its sequence).
  // Mirrors KalmanFitterFunctionImpl, so the DAF picks up the same material
  // between the innermost hit and the perigee as the KF does.
  Propagator referencePropagator;

  Acts::GainMatrixUpdater dafUpdater;
  Acts::MbfSmoother dafSmoother;

  bool multipleScattering = true;
  bool energyLoss = true;
  bool useDirectNavigator = true;
  bool useKfSequenceHandling = false;
  Acts::FreeToBoundCorrection freeToBoundCorrection;
  Acts::AnnealingUtility::Config annealingConfig;
  std::size_t iterationsPerTemperature = 1;
  std::size_t maxIterations = 100;

  /// Drift-circle initial DAF weights on straw surfaces, as a multiple of half
  /// the straw's radial cell spacing; 0 keeps the uniform 1/N start. 3
  /// reproduces GenFit's rMax = 1.5 x cell spacing for the Belle2 CDC. See
  /// DafActor::strawInitialWeightRadiusFactor.
  double strawInitialWeightRadiusFactor = 0.;

  /// Combine mirror hypotheses by moment matching rather than the default
  /// information-form sum. See Acts::detail::momentMatchedDafWeightedCombiner.
  bool useMomentMatchedCombiner = false;

  /// Inflation applied to the re-seed covariance between annealing passes;
  /// 1 keeps the previous behaviour. See
  /// DafFitterOptions::reseedCovarianceScaling.
  double reseedCovarianceScaling = 1.;

  /// Which covariance the per-candidate weight chi2 divides by. Full is the
  /// previous behaviour; MeasurementOnly is GenFit's; Unbiased is the correct
  /// one for a biased smoothed state. See detail::DafChiSquareDenominator.
  /// GenFit's convergence tests; both tolerances default to 0 = disabled.
  /// See DafFitterOptions for what each one means and why reproducing GenFit
  /// means pValueConvergenceTolerance = 1 rather than something small.
  double weightConvergenceTolerance = 0.;
  std::size_t minAnnealingIterations = 1;
  double pValueConvergenceTolerance = 0.;
  double pValueMinimum = 0.;
  std::size_t pValueMinIterations = 5;

  /// GenFit's blowUpCov trio: reset off-diagonals, and clip the blown-up
  /// variances. Angles get their own ceiling so a large scaling cannot push
  /// them past the branch cut.
  bool reseedResetOffDiagonals = false;
  double reseedVarianceCeiling = 0.;
  double reseedAngleVarianceCeiling = 0.;
  /// Same blow-up applied to the pass-0 seed (GenFit's iteration 1).
  bool blowUpInitialSeed = false;

  /// GenFit's smoothing: backward filter combined with the forward one.
  bool useTwoFilterSmoothing = false;
  double twoFilterBlowUpFactor = 1000.;

  /// GenFit's prepareTrack: piecewise reference, re-linearised around the
  /// previous pass's smoothed states. No effect without referenceLinearization.
  bool referencePiecewise = false;
  bool referenceAnchorSmoothed = false;
  bool referencePiecewiseSelfCheck = false;

  Acts::detail::DafChiSquareDenominator chiSquareDenominator =
      Acts::detail::DafChiSquareDenominator::Full;

  /// Linearise about a reference trajectory (GenFit's KalmanFitterRefTrack)
  /// instead of the running filtered state. See
  /// DafFitterOptions::referenceLinearization.
  bool referenceLinearization = false;

  /// Score the weights on the reference-calibrated measurements rather than
  /// re-calibrating them against the smoothed state, as GenFit's DAF does. Only
  /// meaningful alongside referenceLinearization. See
  /// DafFitterOptions::referenceWeightCalibration.
  bool referenceWeightCalibration = true;

  /// Factor on the seed covariance the reference filter starts each pass from.
  /// Not GenFit's blowUpFactor -- see

  std::vector<const Acts::Surface*> materialSurfaces;

  // Kept so the direct-navigation branch can work out the seed curvature; the
  // stepper owns its own copy.
  std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;

  IndexSourceLink::SurfaceAccessor slSurfaceAccessor;

  /// Signed transverse radius of the seed helix, in ACTS length units. Zero
  /// when it cannot be determined, which makes the sequence preparation fall
  /// back to straight-line paths. Mirrors KalmanFitterFunctionImpl.
  double seedSignedRadius(const TrackParameters& params,
                          const GeneralFitterOptions& options) const {
    if (magneticField == nullptr) {
      return 0.;
    }
    auto cache = magneticField->makeCache(options.magFieldContext);
    auto field =
        magneticField->getField(params.position(options.geoContext), cache);
    if (!field.ok()) {
      return 0.;
    }
    const double bz = field.value().z();
    const double qOverP = params.parameters()[Acts::eBoundQOverP];
    if (std::abs(bz) < 1e-12 || std::abs(qOverP) < 1e-12) {
      return 0.;
    }
    return std::sin(params.parameters()[Acts::eBoundTheta]) / (qOverP * bz);
  }

  DafFitterFunctionImpl(Fitter&& f, DirectFitter&& df, Propagator&& refProp,
                        const Acts::TrackingGeometry& trkGeo,
                        std::shared_ptr<const Acts::MagneticFieldProvider> bField)
      : fitter(std::move(f)),
        directFitter(std::move(df)),
        referencePropagator(std::move(refProp)),
        magneticField(std::move(bField)),
        slSurfaceAccessor{trkGeo} {
    trkGeo.visitSurfaces(
        [this](const Acts::Surface* srf) {
          if (srf != nullptr && srf->surfaceMaterial() != nullptr &&
              materialVolumeAccepted(
                  static_cast<unsigned>(srf->geometryId().volume()))) {
            materialSurfaces.push_back(srf);
          }
        },
        false);
  }

  template <typename calibrator_t>
  auto makeDafOptions(const GeneralFitterOptions& options,
                      const calibrator_t& calibrator) const {
    Acts::DafFitterExtensions<Acts::VectorMultiTrajectory> extensions;
    extensions.calibrator.template connect<&calibrator_t::calibrate>(
        &calibrator);
    using Denominator = Acts::detail::DafChiSquareDenominator;
    // The chi2 and its Gaussian prefactor are connected together, with the same
    // denominator, in every branch: they are the exponent and the normalisation
    // of one density, and connecting them with different S would silently stop
    // the ratio being one.
    switch (chiSquareDenominator) {
      case Denominator::MeasurementOnly:
        extensions.chiSquareCalculator
            .template connect<&Acts::detail::defaultDafChiSquareCalculator<
                Acts::VectorMultiTrajectory, Denominator::MeasurementOnly>>();
        extensions.logNormalization
            .template connect<&Acts::detail::defaultDafLogNormalization<
                Acts::VectorMultiTrajectory, Denominator::MeasurementOnly>>();
        break;
      case Denominator::Unbiased:
        extensions.chiSquareCalculator
            .template connect<&Acts::detail::defaultDafChiSquareCalculator<
                Acts::VectorMultiTrajectory, Denominator::Unbiased>>();
        extensions.logNormalization
            .template connect<&Acts::detail::defaultDafLogNormalization<
                Acts::VectorMultiTrajectory, Denominator::Unbiased>>();
        break;
      case Denominator::Full:
        extensions.chiSquareCalculator
            .template connect<&Acts::detail::defaultDafChiSquareCalculator<
                Acts::VectorMultiTrajectory, Denominator::Full>>();
        extensions.logNormalization
            .template connect<&Acts::detail::defaultDafLogNormalization<
                Acts::VectorMultiTrajectory, Denominator::Full>>();
        break;
    }
    if (useMomentMatchedCombiner) {
      extensions.weightedCombiner
          .template connect<&Acts::detail::momentMatchedDafWeightedCombiner<
              Acts::VectorMultiTrajectory>>();
    } else {
      extensions.weightedCombiner
          .template connect<&Acts::detail::defaultDafWeightedCombiner<
              Acts::VectorMultiTrajectory>>();
    }
    extensions.updater.template connect<
        &Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(
        &dafUpdater);
    extensions.smoother.template connect<
        &Acts::MbfSmoother::operator()<Acts::VectorMultiTrajectory>>(
        &dafSmoother);
    extensions.surfaceAccessor
        .connect<&IndexSourceLink::SurfaceAccessor::operator()>(
            &slSurfaceAccessor);

    Acts::DafFitterOptions<Acts::VectorMultiTrajectory> dafOptions(
        options.geoContext, options.magFieldContext, options.calibrationContext,
        extensions, options.propOptions, options.referenceSurface,
        multipleScattering, energyLoss, freeToBoundCorrection, annealingConfig,
        iterationsPerTemperature, maxIterations);
    dafOptions.referenceExtrapolationSeconds =
        options.referenceExtrapolationSeconds;

    dafOptions.referenceSurfaceStrategy =
        Acts::TrackExtrapolationStrategy::first;
    dafOptions.strawInitialWeightRadiusFactor =
        strawInitialWeightRadiusFactor;
    dafOptions.reseedCovarianceScaling = reseedCovarianceScaling;
    dafOptions.referenceLinearization = referenceLinearization;
    dafOptions.referenceWeightCalibration = referenceWeightCalibration;
    dafOptions.useUnbiasedSmoothedState = useUnbiasedSmoothedState();
    dafOptions.useCandidateNormalization = useCandidateNormalization();
    dafOptions.weightConvergenceTolerance = weightConvergenceTolerance;
    dafOptions.minIterations = minAnnealingIterations;
    dafOptions.pValueConvergenceTolerance = pValueConvergenceTolerance;
    dafOptions.pValueMinimum = pValueMinimum;
    dafOptions.pValueMinIterations = pValueMinIterations;
    dafOptions.referencePiecewise = referencePiecewise;
    dafOptions.referenceAnchorSmoothed = referenceAnchorSmoothed;
    dafOptions.referencePiecewiseSelfCheck = referencePiecewiseSelfCheck;
    dafOptions.useTwoFilterSmoothing = useTwoFilterSmoothing;
    dafOptions.twoFilterBlowUpFactor = twoFilterBlowUpFactor;
    dafOptions.reseedResetOffDiagonals = reseedResetOffDiagonals;
    dafOptions.reseedVarianceCeiling = reseedVarianceCeiling;
    dafOptions.blowUpInitialSeed = blowUpInitialSeed;
    if (reseedAngleVarianceCeiling > 0.) {
      dafOptions.reseedVarianceCeilings[Acts::eBoundPhi] =
          reseedAngleVarianceCeiling;
      dafOptions.reseedVarianceCeilings[Acts::eBoundTheta] =
          reseedAngleVarianceCeiling;
    }
    if (stallStepsOverride() > 0) {
      dafOptions.stallStepLimit = stallStepsOverride();
    }

    return dafOptions;
  }

  TrackFitterResult operator()(
      const std::unordered_map<const Acts::Surface*, Acts::DafMeasurementGroup>&
          measurementGroups,
      const std::vector<const Acts::Surface*>& surfaceSequence,
      const TrackParameters& initialParameters,
      const GeneralFitterOptions& options,
      const MeasurementCalibratorAdapter& calibrator, TrackContainer& tracks,
      Acts::DafCandidateMap* finalCandidateState,
      Acts::DafCandidateTrace* candidateTrace,
      Acts::DafFitStatus* fitStatus) const override {
    auto dafOptions = makeDafOptions(options, calibrator);
    dafOptions.finalCandidateState = finalCandidateState;
    dafOptions.candidateTrace = candidateTrace;
    dafOptions.fitStatus = fitStatus;
    if (useDirectNavigator) {
      // The DirectNavigator cannot target the reference (perigee) surface,
      // which lies outside its sequence: fit without one, then extrapolate
      // the finished track with the geometric-Navigator referencePropagator
      // -- exactly as KalmanFitterFunctionImpl does.
      dafOptions.referenceSurface = nullptr;

      auto runDirectFit =
          [&](const std::unordered_map<const Acts::Surface*,
                                       Acts::DafMeasurementGroup>& groups,
              const std::vector<const Acts::Surface*>& sequence)
          -> TrackFitterResult {
        auto result = directFitter.fit(groups, initialParameters,
                                       dafOptions, sequence, tracks);
        if (result.ok() && options.referenceSurface != nullptr) {
          // The perigee lies inside the innermost measurement, so this
          // extrapolation crosses the CDC inner wall, the SVD, the PXD and the
          // beam pipe. Without a MaterialInteractor the covariance is
          // transported through all of it with no multiple scattering added,
          // and the reported perigee covariance comes out too small -- most
          // visibly in phi, whose fit uncertainty is the smallest of the six so
          // the missing term dominates it. Measured 2026-08-22: the implied
          // missing term scales as 1/p with a constant 3.3 mrad*GeV, i.e.
          // Highland for ~7% X0, and adding it in quadrature reproduces
          // GenFit's sigma_phi exactly.
          //
          // Scattering and energy loss follow the fit's own switches, so a
          // no-material fit does not silently acquire material here.
          // ACTS_O2_ENERGY_LOSS overrides the energy loss alone (see
          // extrapolationEnergyLoss).
          Propagator::Options<Acts::ActorList<Acts::MaterialInteractor>>
              extrapolationOptions(options.geoContext, options.magFieldContext);
          auto& matInteractor = extrapolationOptions.actorList
                                    .template get<Acts::MaterialInteractor>();
          matInteractor.multipleScattering = multipleScattering;
          matInteractor.energyLoss = extrapolationEnergyLoss(energyLoss);
          // Wall-clock time of this extrapolation, for a caller timing the
          // fit without it (GeneralFitterOptions::referenceExtrapolationSeconds).
          const auto extrapolationStart = std::chrono::steady_clock::now();
          auto addExtrapolationTime = [&]() {
            if (options.referenceExtrapolationSeconds != nullptr) {
              *options.referenceExtrapolationSeconds +=
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - extrapolationStart)
                      .count();
            }
          };
          if (o2MaterialDumpDir() != nullptr) {
            // Same extrapolation, with the material recorded and written out.
            matInteractor.recordInteractions = true;
            auto dumped = extrapolateAndDumpMaterial(
                result.value(), *options.referenceSurface, referencePropagator,
                extrapolationOptions, dafOptions.referenceSurfaceStrategy,
                options.geoContext);
            addExtrapolationTime();
            if (!dumped.ok()) {
              return dumped.error();
            }
          } else {
            auto extrapolationResult = Acts::extrapolateTrackToReferenceSurface(
                result.value(), *options.referenceSurface, referencePropagator,
                extrapolationOptions, dafOptions.referenceSurfaceStrategy);
            addExtrapolationTime();
            if (!extrapolationResult.ok()) {
              return extrapolationResult.error();
            }
          }
        }
        return result;
      };

      if (useKfSequenceHandling) {
        // KF-equivalence mode: run the sequence through the exact same
        // preparation as KalmanFitterFunctionImpl (helix-path sort from the
        // seed, material-surface injection, path limit sized from the returned
        // span) so both fitters traverse the identical DirectNavigator
        // sequence.
        std::vector<const Acts::Surface*> sequence = surfaceSequence;
        // Filled by the reachability cut (ACTS_DNS_MAXREACH) with the hit
        // surfaces the seed helix cannot reach; the branch below erases their
        // measurement groups, without which DafActor never finishes a pass.
        std::vector<const Acts::Surface*> dropped;
        const double sequenceSpan = prepareDirectNavigationSequence(
            options.geoContext, initialParameters.position(options.geoContext),
            initialParameters.direction(), sequence, materialSurfaces,
            seedSignedRadius(initialParameters, options), &dropped);
        // Sized from the sequence rather than fixed: a constant 1.5 m
        // truncates curlers, whose arc length grows as the momentum falls.
        dafOptions.propagatorPlainOptions.pathLimit =
            pathLimitScale() *
            std::max(1.5 * Acts::UnitConstants::m,
                     1.5 * sequenceSpan + 0.5 * Acts::UnitConstants::m);
        // ...and let that limit actually govern: Acts' loop protection would
        // otherwise clamp the pass to half a helix turn regardless. See
        // loopProtection() above for why that is wrong here and what it costs.
        dafOptions.propagatorPlainOptions.loopProtection = loopProtection();
        if (maxStepsOverride() > 0) {
          dafOptions.propagatorPlainOptions.maxSteps = maxStepsOverride();
        }

        // Per-pass re-linearisation of the navigation geometry: every DAF pass
        // rebuilds the sequence from its own re-seeded parameters instead of
        // reusing this one, which is frozen at the original seed. GenFit's
        // KalmanFitterRefTrack does the equivalent via setRefitAll. Installed
        // only in this branch so that pass 1 and every later pass go through
        // the identical preparation -- with the raw sequence below, a rebuild
        // would suddenly sort it and inject material from pass 2 onwards.
        //
        // The reachability cut is deliberately NOT passed a `dropped` vector
        // here: on a later pass a hit surface that this seed cannot reach is
        // still a real hit, and dropping it would make DafActor's termination
        // condition unsatisfiable. The fitter checks for exactly that and falls
        // back to the previous pass's sequence.
        //
        // pathLimit stays as computed above, from the first sequence's span. A
        // rebuilt sequence spans a shorter arc as the seed improves, so this is
        // conservative rather than truncating; it is not re-derived because
        // plainOptions are consumed before the pass loop starts.
        if (rebuildSequencePerPass()) {
          dafOptions.rebuildSequence =
              [this, &options,
               &surfaceSequence](const Acts::BoundTrackParameters& passSeed) {
                std::vector<const Acts::Surface*> rebuilt = surfaceSequence;
                prepareDirectNavigationSequence(
                    options.geoContext, passSeed.position(options.geoContext),
                    passSeed.direction(), rebuilt, materialSurfaces,
                    seedSignedRadius(passSeed, options));
                return rebuilt;
              };
        }
        if (resortSequencePerPass()) {
          dafOptions.resortSequence =
              [this, &options](
                  const std::vector<const Acts::Surface*>& previousSequence,
                  const std::vector<
                      Acts::DafFitterOptions<
                          Acts::VectorMultiTrajectory>::ReachedSurface>&
                      reached) {
                std::vector<ReachedTrajectoryPoint> points;
                points.reserve(reached.size());
                for (const auto& r : reached) {
                  points.push_back({r.surface, r.pathLength,
                                    r.parameters.position(options.geoContext),
                                    r.parameters.direction(),
                                    seedSignedRadius(r.parameters, options)});
                }
                return resortDirectNavigationSequence(options.geoContext,
                                                      previousSequence, points);
              };
        }

        if (dropped.empty()) {
          return runDirectFit(measurementGroups, sequence);
        }
        // The reachability cut removed hit surfaces from the sequence, so their
        // groups have to go too: DafActor ends a pass on
        // measurementStates == inputMeasurementGroups.size(), and a group that
        // is never visited would leave that unsatisfied for the whole
        // propagation -- trading the step burn we are trying to avoid for a
        // path-limit burn.
        auto reachableGroups = measurementGroups;
        for (const Acts::Surface* srf : dropped) {
          reachableGroups.erase(srf);
        }
        if (reachableGroups.empty()) {
          return Acts::DafFitterError::NoMeasurementFound;
        }
        return runDirectFit(reachableGroups, sequence);
      }

      // Default: feed the DirectNavigator the surfaces in proto-track
      // traversal order, exactly as DafTrackFittingAlgorithm assembled them.
      // That order comes from the actual pattern-recognized hit sequence and
      // is the only one reliably monotonic along the real trajectory. Sorting
      // by geometryId().layer() (as a previous version did) reorders overlap
      // and same-layer surfaces, and any segment that is not radially
      // monotonic, into an order the navigator then has to backtrack through
      // -- which for low-momentum curlers degenerates into an oscillating
      // target search that burns the entire propagation step budget and fails
      // the fit. (For a purely radial silicon sequence the two orders
      // coincide, so this only bites once non-monotonic subdetectors such as
      // the CDC are involved.)
      return runDirectFit(measurementGroups, surfaceSequence);
    }
    return fitter.fit(measurementGroups, initialParameters, dafOptions, tracks);
  }
};

}  // namespace

std::shared_ptr<ActsExamples::DafFitterFunction>
ActsExamples::makeDafFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    bool multipleScattering, bool energyLoss,
    Acts::FreeToBoundCorrection freeToBoundCorrection,
    std::vector<double> annealingTemperatures, double annealingCutOff,
    std::size_t iterationsPerTemperature, std::size_t maxIterations,
    bool useDirectNavigator, bool useKfSequenceHandling,
    double strawInitialWeightRadiusFactor, bool useMomentMatchedCombiner,
    double reseedCovarianceScaling,
    Acts::detail::DafChiSquareDenominator chiSquareDenominator,
    bool annealingCutOffPerCandidate,
    bool referenceLinearization,
    bool referenceWeightCalibration,
    double weightConvergenceTolerance, std::size_t minAnnealingIterations,
    double pValueConvergenceTolerance, double pValueMinimum,
    std::size_t pValueMinIterations, bool reseedResetOffDiagonals,
    double reseedVarianceCeiling, double reseedAngleVarianceCeiling,
    bool useTwoFilterSmoothing, double twoFilterBlowUpFactor,
    bool referencePiecewise, bool referenceAnchorSmoothed,
    bool referencePiecewiseSelfCheck, bool blowUpInitialSeed,
    const Acts::Logger& logger) {
  // Stepper should be copied into the fitters. Keep a handle on the field
  // first: the direct-navigation branch needs it to size the seed curvature.
  auto fieldForCurvature = magneticField;
  const Stepper stepper(std::move(magneticField));

  // Standard fitter
  const auto& geo = *trackingGeometry;
  Acts::Navigator::Config cfg{std::move(trackingGeometry)};
  cfg.resolvePassive = false;
  cfg.resolveMaterial = true;
  cfg.resolveSensitive = true;
  Acts::Navigator navigator(cfg, logger.cloneWithSuffix("Navigator"));
  Propagator propagator(stepper, std::move(navigator),
                        logger.cloneWithSuffix("Propagator"));
  Fitter trackFitter(std::move(propagator), logger.cloneWithSuffix("Fitter"));

  // Direct fitter
  Acts::DirectNavigator directNavigator{
      logger.cloneWithSuffix("DirectNavigator")};
  DirectPropagator directPropagator(stepper, std::move(directNavigator),
                                    logger.cloneWithSuffix("DirectPropagator"));
  DirectFitter directTrackFitter(std::move(directPropagator),
                                 logger.cloneWithSuffix("DirectFitter"));

  // Geometric-navigation propagator for the post-fit reference-surface
  // extrapolation of direct-navigation fits.
  Acts::Navigator refNavigator(cfg, logger.cloneWithSuffix("RefNavigator"));
  Propagator refPropagator(stepper, std::move(refNavigator),
                           logger.cloneWithSuffix("RefPropagator"));

  // build the fitter function. owns the fitter object.
  auto fitterFunction = std::make_shared<DafFitterFunctionImpl>(
      std::move(trackFitter), std::move(directTrackFitter),
      std::move(refPropagator), geo, std::move(fieldForCurvature));
  fitterFunction->multipleScattering = multipleScattering;
  fitterFunction->energyLoss = energyLoss;
  fitterFunction->useDirectNavigator = useDirectNavigator;
  fitterFunction->useKfSequenceHandling = useKfSequenceHandling;
  fitterFunction->freeToBoundCorrection = freeToBoundCorrection;
  fitterFunction->annealingConfig.setOfTemperatures =
      std::move(annealingTemperatures);
  fitterFunction->annealingConfig.cutOff = annealingCutOff;
  fitterFunction->annealingConfig.cutOffPerCandidate =
      annealingCutOffPerCandidate;
  fitterFunction->iterationsPerTemperature = iterationsPerTemperature;
  fitterFunction->maxIterations = maxIterations;
  fitterFunction->strawInitialWeightRadiusFactor =
      strawInitialWeightRadiusFactor;
  fitterFunction->weightConvergenceTolerance = weightConvergenceTolerance;
  fitterFunction->minAnnealingIterations = minAnnealingIterations;
  fitterFunction->pValueConvergenceTolerance = pValueConvergenceTolerance;
  fitterFunction->pValueMinimum = pValueMinimum;
  fitterFunction->pValueMinIterations = pValueMinIterations;
  fitterFunction->reseedResetOffDiagonals = reseedResetOffDiagonals;
  fitterFunction->reseedVarianceCeiling = reseedVarianceCeiling;
  fitterFunction->reseedAngleVarianceCeiling = reseedAngleVarianceCeiling;
  fitterFunction->useTwoFilterSmoothing = useTwoFilterSmoothing;
  fitterFunction->twoFilterBlowUpFactor = twoFilterBlowUpFactor;
  fitterFunction->referencePiecewise = referencePiecewise;
  fitterFunction->referenceAnchorSmoothed = referenceAnchorSmoothed;
  fitterFunction->referencePiecewiseSelfCheck = referencePiecewiseSelfCheck;
  fitterFunction->blowUpInitialSeed = blowUpInitialSeed;
  fitterFunction->useMomentMatchedCombiner = useMomentMatchedCombiner;
  fitterFunction->reseedCovarianceScaling = reseedCovarianceScaling;
  fitterFunction->chiSquareDenominator = chiSquareDenominator;
  fitterFunction->referenceLinearization = referenceLinearization;
  fitterFunction->referenceWeightCalibration = referenceWeightCalibration;

  return fitterFunction;
}

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
#include <cstdlib>
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
double pathLimitScale() {
  static const double value = [] {
    const char* env = std::getenv("ACTS_DAF_PATH_SCALE");
    return env == nullptr ? 1.0 : std::strtod(env, nullptr);
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

unsigned maxStepsOverride() {
  static const unsigned value = [] {
    const char* env = std::getenv("ACTS_MAX_STEPS");
    return env == nullptr ? 0u
                          : static_cast<unsigned>(std::strtoul(env, nullptr, 10));
  }();
  return value;
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
  /// DafFitterOptions::referenceStartCovarianceScaling for why 1e3 breaks here.
  double referenceStartCovarianceScaling = 1.;

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
    extensions.chiSquareCalculator
        .template connect<&Acts::detail::defaultDafChiSquareCalculator<
            Acts::VectorMultiTrajectory>>();
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

    dafOptions.referenceSurfaceStrategy =
        Acts::TrackExtrapolationStrategy::first;
    dafOptions.strawInitialWeightRadiusFactor =
        strawInitialWeightRadiusFactor;
    dafOptions.reseedCovarianceScaling = reseedCovarianceScaling;
    dafOptions.referenceLinearization = referenceLinearization;
    dafOptions.referenceWeightCalibration = referenceWeightCalibration;
    dafOptions.referenceStartCovarianceScaling =
        referenceStartCovarianceScaling;
    dafOptions.useUnbiasedSmoothedState = useUnbiasedSmoothedState();
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
      Acts::DafCandidateTrace* candidateTrace) const override {
    auto dafOptions = makeDafOptions(options, calibrator);
    dafOptions.finalCandidateState = finalCandidateState;
    dafOptions.candidateTrace = candidateTrace;
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
          Propagator::Options<> extrapolationOptions(options.geoContext,
                                                     options.magFieldContext);
          auto extrapolationResult = Acts::extrapolateTrackToReferenceSurface(
              result.value(), *options.referenceSurface, referencePropagator,
              extrapolationOptions, dafOptions.referenceSurfaceStrategy);
          if (!extrapolationResult.ok()) {
            return extrapolationResult.error();
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
    double reseedCovarianceScaling, bool referenceLinearization,
    bool referenceWeightCalibration, double referenceStartCovarianceScaling,
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
  fitterFunction->iterationsPerTemperature = iterationsPerTemperature;
  fitterFunction->maxIterations = maxIterations;
  fitterFunction->strawInitialWeightRadiusFactor =
      strawInitialWeightRadiusFactor;
  fitterFunction->useMomentMatchedCombiner = useMomentMatchedCombiner;
  fitterFunction->reseedCovarianceScaling = reseedCovarianceScaling;
  fitterFunction->referenceLinearization = referenceLinearization;
  fitterFunction->referenceWeightCalibration = referenceWeightCalibration;
  fitterFunction->referenceStartCovarianceScaling =
      referenceStartCovarianceScaling;

  return fitterFunction;
}

// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/TrackContainer.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/detail/CorrectedTransformationFreeToBound.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/SympyStepper.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/MbfSmoother.hpp"
#include "Acts/Utilities/Intersection.hpp"
#include "Acts/Utilities/Delegate.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/TrackFitting/DirectNavigationSequence.hpp"
#include "ActsExamples/TrackFitting/RefittingCalibrator.hpp"
#include "ActsExamples/TrackFitting/TrackFitterFunction.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace {

using Stepper = Acts::SympyStepper;
using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
using Fitter = Acts::KalmanFitter<Propagator, Acts::VectorMultiTrajectory>;
using DirectPropagator = Acts::Propagator<Stepper, Acts::DirectNavigator>;
using DirectFitter =
    Acts::KalmanFitter<DirectPropagator, Acts::VectorMultiTrajectory>;

using TrackContainer =
    Acts::TrackContainer<Acts::VectorTrackContainer,
                         Acts::VectorMultiTrajectory, std::shared_ptr>;

struct SimpleReverseFilteringLogic {
  double momentumThreshold = 0;

  bool doBackwardFiltering(
      Acts::VectorMultiTrajectory::ConstTrackStateProxy trackState) const {
    auto momentum = std::abs(1 / trackState.filtered()[Acts::eBoundQOverP]);
    return (momentum <= momentumThreshold);
  }
};

struct SimpleOutlierFinder {
  double chi2Cut = std::numeric_limits<double>::infinity();

  bool isOutlier(
      Acts::VectorMultiTrajectory::ConstTrackStateProxy trackState) const {
    double chi2 = Acts::calculatePredictedChi2(trackState);
    return chi2 > chi2Cut;
  }
};


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

unsigned maxStepsOverride() {
  static const unsigned value = [] {
    const char* env = std::getenv("ACTS_MAX_STEPS");
    return env == nullptr ? 0u
                          : static_cast<unsigned>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// Restore Acts' loop protection on the direct-navigation fit
/// (ACTS_LOOP_PROTECTION=1), which this branch otherwise switches off. The DAF
/// reads the same variable; DafFitterFunction.cpp carries the full rationale
/// and the measurements. In short: loop protection caps the propagation at
/// half a helix turn (loopFraction = 0.5), which overrides the pathLimit sized
/// from the sequence span below, and KalmanFitter.hpp:355 sets its own
/// result-side aborter to that limit outright. Any track whose hits span more
/// than half a revolution is silently truncated there.
///
/// Path limiting is not lost: `Propagator::makeState` appends a
/// PathLimitReached aborter carrying `options.pathLimit` (Propagator.ipp:255).
bool loopProtection() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_LOOP_PROTECTION");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}
}  // namespace

using namespace ActsExamples;

struct KalmanFitterFunctionImpl final : public TrackFitterFunction {
  Fitter fitter;
  DirectFitter directFitter;
  // Plain geometric-Navigator propagator used only for the final
  // extrapolation to the reference surface after a direct-navigation fit
  // (the DirectNavigator cannot target a surface outside its sequence).
  Propagator referencePropagator;

  Acts::GainMatrixUpdater kfUpdater;
  Acts::MbfSmoother kfSmoother;
  SimpleReverseFilteringLogic reverseFilteringLogic;
  double reverseFilteringCovarianceScaling = 100.0;
  SimpleOutlierFinder outlierFinder;

  bool multipleScattering = false;
  bool energyLoss = false;
  Acts::FreeToBoundCorrection freeToBoundCorrection;

  bool directNavigation = false;

  std::vector<const Acts::Surface*> materialSurfaces;

  // Kept so the direct-navigation branch can work out the seed curvature; the
  // stepper owns its own copy.
  std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;

  IndexSourceLink::SurfaceAccessor slSurfaceAccessor;

  /// Signed transverse radius of the seed helix, in ACTS length units. Zero
  /// when it cannot be determined, which makes the sequence preparation fall
  /// back to straight-line paths.
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
    // pT = sin(theta)/|q/p| and R = pT/(qB), so R_signed = sin(theta)/((q/p)*Bz)
    return std::sin(params.parameters()[Acts::eBoundTheta]) / (qOverP * bz);
  }

  KalmanFitterFunctionImpl(Fitter&& f, DirectFitter&& df, Propagator&& refProp,
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
  auto makeKfOptions(const GeneralFitterOptions& options,
                     const calibrator_t& calibrator) const {
    Acts::KalmanFitterExtensions<Acts::VectorMultiTrajectory> extensions;
    extensions.updater.connect<
        &Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(
        &kfUpdater);
    extensions.smoother
        .connect<&Acts::MbfSmoother::operator()<Acts::VectorMultiTrajectory>>(
            &kfSmoother);
    extensions.reverseFilteringLogic
        .connect<&SimpleReverseFilteringLogic::doBackwardFiltering>(
            &reverseFilteringLogic);
    extensions.outlierFinder.connect<&SimpleOutlierFinder::isOutlier>(
        &outlierFinder);

    Acts::KalmanFitterOptions<Acts::VectorMultiTrajectory> kfOptions(
        options.geoContext, options.magFieldContext, options.calibrationContext,
        extensions, options.propOptions, options.referenceSurface);

    kfOptions.referenceSurfaceStrategy =
        Acts::TrackExtrapolationStrategy::first;
    kfOptions.multipleScattering = multipleScattering;
    kfOptions.energyLoss = energyLoss;
    kfOptions.freeToBoundCorrection = freeToBoundCorrection;
    kfOptions.extensions.calibrator.connect<&calibrator_t::calibrate>(
        &calibrator);
    kfOptions.reverseFilteringCovarianceScaling =
        reverseFilteringCovarianceScaling;

    if (options.doRefit) {
      kfOptions.extensions.surfaceAccessor
          .connect<&RefittingCalibrator::accessSurface>();
    } else {
      kfOptions.extensions.surfaceAccessor
          .connect<&IndexSourceLink::SurfaceAccessor::operator()>(
              &slSurfaceAccessor);
    }

    return kfOptions;
  }

  /// Run the direct-navigation fit end to end: prepare the surface sequence,
  /// size the propagator limits from the sequence actually being traversed,
  /// fit, then extrapolate to the reference surface through material.
  ///
  /// Shared by the initial fit and the refit so an iterated fit repeats the
  /// identical procedure. Before this was factored out the refit ran on plain
  /// PropagatorPlainOptions, which differs in two ways that each cost hits or
  /// accuracy: loop protection is back on there (capping the pass at half a
  /// helix turn, truncating curlers -- see loopProtection() above), and the
  /// reference surface was left set inside the direct fit, asking the
  /// DirectNavigator to target a perigee outside its own sequence instead of
  /// extrapolating to it afterwards with a MaterialInteractor.
  ///
  /// @param surfaceSequence taken by value: preparation reorders it in place.
  TrackFitterResult fitDirect(
      const std::vector<Acts::SourceLink>& sourceLinks,
      const TrackParameters& initialParameters,
      const GeneralFitterOptions& options,
      const Acts::KalmanFitterOptions<Acts::VectorMultiTrajectory>& kfOptions,
      std::vector<const Acts::Surface*> surfaceSequence,
      TrackContainer& tracks) const {
    const double sequenceSpan = prepareDirectNavigationSequence(
        options.geoContext, initialParameters.position(options.geoContext),
        initialParameters.direction(), surfaceSequence, materialSurfaces,
        seedSignedRadius(initialParameters, options));

    auto directKfOptions = kfOptions;
    directKfOptions.referenceSurface = nullptr;

    // Size the limit from the sequence actually being traversed. A fixed
    // 1.5 m truncates curlers: one turn at 0.13 GeV in 1.5 T is already
    // 1.8 m of arc, so the propagation aborted before reaching the outer
    // hits and the track was dropped by the minMeasurements cut. The 1.5 m
    // floor keeps well-behaved tracks exactly as before.
    directKfOptions.propagatorPlainOptions.pathLimit =
        std::max(1.5 * Acts::UnitConstants::m,
                 1.5 * sequenceSpan + 0.5 * Acts::UnitConstants::m);
    // ...and let that limit actually govern: Acts' loop protection would
    // otherwise clamp the pass to half a helix turn regardless. See
    // loopProtection() above.
    directKfOptions.propagatorPlainOptions.loopProtection = loopProtection();
    if (maxStepsOverride() > 0) {
      directKfOptions.propagatorPlainOptions.maxSteps = maxStepsOverride();
    }

    auto result = directFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                                   initialParameters, directKfOptions,
                                   surfaceSequence, tracks);
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
      Propagator::Options<Acts::ActorList<Acts::MaterialInteractor>>
          extrapolationOptions(options.geoContext, options.magFieldContext);
      auto& matInteractor = extrapolationOptions.actorList
                                .template get<Acts::MaterialInteractor>();
      matInteractor.multipleScattering = multipleScattering;
      matInteractor.energyLoss = energyLoss;
      auto extrapolationResult = Acts::extrapolateTrackToReferenceSurface(
          result.value(), *options.referenceSurface, referencePropagator,
          extrapolationOptions, kfOptions.referenceSurfaceStrategy);
      if (!extrapolationResult.ok()) {
        return extrapolationResult.error();
      }
    }
    return result;
  }

  TrackFitterResult operator()(const std::vector<Acts::SourceLink>& sourceLinks,
                               const TrackParameters& initialParameters,
                               const GeneralFitterOptions& options,
                               const MeasurementCalibratorAdapter& calibrator,
                               TrackContainer& tracks) const override {
    const auto kfOptions = makeKfOptions(options, calibrator);
    if (directNavigation) {
      std::vector<const Acts::Surface*> surfaceSequence;
      surfaceSequence.reserve(sourceLinks.size());
      bool allResolved = true;
      for (const auto& sourceLink : sourceLinks) {
        const Acts::Surface* surface = slSurfaceAccessor(sourceLink);
        if (surface == nullptr) {
          allResolved = false;
          break;
        }
        surfaceSequence.push_back(surface);
      }
      if (allResolved) {
        return fitDirect(sourceLinks, initialParameters, options, kfOptions,
                         std::move(surfaceSequence), tracks);
      }
    }
    return fitter.fit(sourceLinks.begin(), sourceLinks.end(), initialParameters,
                      kfOptions, tracks);
  }

  TrackFitterResult operator()(
      const std::vector<Acts::SourceLink>& sourceLinks,
      const TrackParameters& initialParameters,
      const GeneralFitterOptions& options,
      const RefittingCalibrator& calibrator,
      const std::vector<const Acts::Surface*>& surfaceSequence,
      TrackContainer& tracks) const override {
    const auto kfOptions = makeKfOptions(options, calibrator);
    // The sequence handed in was rebuilt from the input track's own states, so
    // it is already in traversal order and already carries the material
    // surfaces the previous pass stopped at. Running it through the same
    // preparation anyway is what keeps an iterated fit walking the identical
    // sequence every pass: it re-injects anything the previous pass did not
    // record, drops the duplicates that creates, and reorders along the
    // *re-seeded* helix rather than the original one.
    return fitDirect(sourceLinks, initialParameters, options, kfOptions,
                     surfaceSequence, tracks);
  }
};

}  // namespace

std::shared_ptr<TrackFitterFunction> ActsExamples::makeKalmanFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    bool multipleScattering, bool energyLoss,
    double reverseFilteringMomThreshold,
    double reverseFilteringCovarianceScaling,
    Acts::FreeToBoundCorrection freeToBoundCorrection, double chi2Cut,
    bool directNavigation, const Acts::Logger& logger) {
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
  auto fitterFunction = std::make_shared<KalmanFitterFunctionImpl>(
      std::move(trackFitter), std::move(directTrackFitter),
      std::move(refPropagator), geo, std::move(fieldForCurvature));
  fitterFunction->multipleScattering = multipleScattering;
  fitterFunction->energyLoss = energyLoss;
  fitterFunction->reverseFilteringLogic.momentumThreshold =
      reverseFilteringMomThreshold;
  fitterFunction->freeToBoundCorrection = freeToBoundCorrection;
  fitterFunction->reverseFilteringCovarianceScaling =
      reverseFilteringCovarianceScaling;
  fitterFunction->outlierFinder.chi2Cut = chi2Cut;
  fitterFunction->directNavigation = directNavigation;

  return fitterFunction;
}

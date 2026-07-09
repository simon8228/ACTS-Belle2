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
#include "ActsExamples/TrackFitting/RefittingCalibrator.hpp"
#include "ActsExamples/TrackFitting/TrackFitterFunction.hpp"

#include <algorithm>
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

  IndexSourceLink::SurfaceAccessor slSurfaceAccessor;

  KalmanFitterFunctionImpl(Fitter&& f, DirectFitter&& df, Propagator&& refProp,
                           const Acts::TrackingGeometry& trkGeo)
      : fitter(std::move(f)),
        directFitter(std::move(df)),
        referencePropagator(std::move(refProp)),
        slSurfaceAccessor{trkGeo} {
    trkGeo.visitSurfaces(
        [this](const Acts::Surface* srf) {
          if (srf != nullptr && srf->surfaceMaterial() != nullptr) {
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
        const Acts::Vector3 seedPosition =
            initialParameters.position(options.geoContext);
        const Acts::Vector3 seedDirection = initialParameters.direction();
        auto pathTo = [&](const Acts::Surface* srf) -> double {
          auto multiIntersection =
              srf->intersect(options.geoContext, seedPosition, seedDirection,
                             Acts::BoundaryTolerance::Infinite());
          const auto& intersection = multiIntersection.closestForward();
          if (intersection.status() >= Acts::IntersectionStatus::reachable &&
              intersection.pathLength() > 0) {
            return intersection.pathLength();
          }
          // No forward intersection of the ray (e.g. nearly parallel
          // surface): fall back to the projection of the surface center.
          return (srf->center(options.geoContext) - seedPosition)
              .dot(seedDirection);
        };
        double maxHitPath = 0.;
        for (const Acts::Surface* srf : surfaceSequence) {
          auto hitIntersection =
              srf->intersect(options.geoContext, seedPosition, seedDirection,
                             Acts::BoundaryTolerance::None());
          const auto& hitIx = hitIntersection.closestForward();
          if (hitIx.status() >= Acts::IntersectionStatus::reachable &&
              hitIx.pathLength() > 0) {
            maxHitPath = std::max(maxHitPath, hitIx.pathLength());
          }
        }
        for (const Acts::Surface* srf : materialSurfaces) {
          auto multiIntersection =
              srf->intersect(options.geoContext, seedPosition, seedDirection,
                             Acts::BoundaryTolerance::None());
          const auto& intersection = multiIntersection.closestForward();
          if (intersection.status() >= Acts::IntersectionStatus::reachable &&
              intersection.pathLength() > 0 &&
              intersection.pathLength() < maxHitPath + 1.0) {
            surfaceSequence.push_back(srf);
          }
        }

        std::stable_sort(surfaceSequence.begin(), surfaceSequence.end(),
                         [&](const Acts::Surface* a, const Acts::Surface* b) {
                           return pathTo(a) < pathTo(b);
                         });
        surfaceSequence.erase(
            std::unique(surfaceSequence.begin(), surfaceSequence.end()),
            surfaceSequence.end());

        auto directKfOptions = kfOptions;
        directKfOptions.referenceSurface = nullptr;

        directKfOptions.propagatorPlainOptions.pathLimit =
            1.5 * Acts::UnitConstants::m;

        auto result = directFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                                       initialParameters, directKfOptions,
                                       surfaceSequence, tracks);
        if (result.ok() && options.referenceSurface != nullptr) {
          Propagator::Options<> extrapolationOptions(options.geoContext,
                                                     options.magFieldContext);
          auto extrapolationResult = Acts::extrapolateTrackToReferenceSurface(
              result.value(), *options.referenceSurface, referencePropagator,
              extrapolationOptions, kfOptions.referenceSurfaceStrategy);
          if (!extrapolationResult.ok()) {
            return extrapolationResult.error();
          }
        }
        return result;
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
    return directFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                            initialParameters, kfOptions, surfaceSequence,
                            tracks);
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
  // Stepper should be copied into the fitters
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
      std::move(refPropagator), geo);
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

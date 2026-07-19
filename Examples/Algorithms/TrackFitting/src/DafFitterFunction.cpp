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

  std::vector<const Acts::Surface*> materialSurfaces;

  IndexSourceLink::SurfaceAccessor slSurfaceAccessor;

  DafFitterFunctionImpl(Fitter&& f, DirectFitter&& df, Propagator&& refProp,
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
  auto makeDafOptions(const GeneralFitterOptions& options,
                      const calibrator_t& calibrator) const {
    Acts::DafFitterExtensions<Acts::VectorMultiTrajectory> extensions;
    extensions.calibrator.template connect<&calibrator_t::calibrate>(
        &calibrator);
    extensions.chiSquareCalculator
        .template connect<&Acts::detail::defaultDafChiSquareCalculator<
            Acts::VectorMultiTrajectory>>();
    extensions.weightedCombiner
        .template connect<&Acts::detail::defaultDafWeightedCombiner<
            Acts::VectorMultiTrajectory>>();
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

    return dafOptions;
  }

  TrackFitterResult operator()(
      const std::unordered_map<const Acts::Surface*, Acts::DafMeasurementGroup>&
          measurementGroups,
      const std::vector<const Acts::Surface*>& surfaceSequence,
      const TrackParameters& initialParameters,
      const GeneralFitterOptions& options,
      const MeasurementCalibratorAdapter& calibrator, TrackContainer& tracks,
      Acts::DafCandidateMap* finalCandidateState) const override {
    auto dafOptions = makeDafOptions(options, calibrator);
    dafOptions.finalCandidateState = finalCandidateState;
    if (useDirectNavigator) {
      // The DirectNavigator cannot target the reference (perigee) surface,
      // which lies outside its sequence: fit without one, then extrapolate
      // the finished track with the geometric-Navigator referencePropagator
      // -- exactly as KalmanFitterFunctionImpl does.
      dafOptions.referenceSurface = nullptr;

      auto runDirectFit =
          [&](const std::vector<const Acts::Surface*>& sequence)
          -> TrackFitterResult {
        auto result = directFitter.fit(measurementGroups, initialParameters,
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
        // preparation as KalmanFitterFunctionImpl (straight-line-path sort
        // from the seed, material-surface injection, 1.5 m path limit) so
        // both fitters traverse the identical DirectNavigator sequence.
        std::vector<const Acts::Surface*> sequence = surfaceSequence;
        prepareDirectNavigationSequence(
            options.geoContext, initialParameters.position(options.geoContext),
            initialParameters.direction(), sequence, materialSurfaces);
        dafOptions.propagatorPlainOptions.pathLimit =
            1.5 * Acts::UnitConstants::m;
        return runDirectFit(sequence);
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
      return runDirectFit(surfaceSequence);
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
    const Acts::Logger& logger) {
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
  auto fitterFunction = std::make_shared<DafFitterFunctionImpl>(
      std::move(trackFitter), std::move(directTrackFitter),
      std::move(refPropagator), geo);
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

  return fitterFunction;
}

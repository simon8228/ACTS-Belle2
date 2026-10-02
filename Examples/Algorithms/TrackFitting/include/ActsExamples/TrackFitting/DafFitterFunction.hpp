// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/detail/CorrectedTransformationFreeToBound.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/DafChiSquareCalculator.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/TrackFitting/TrackFitterFunction.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ActsExamples {

/// Fit function for the Deterministic Annealing Filter.
///
/// Unlike @ref TrackFitterFunction, this takes competing measurement
/// candidates pre-grouped by surface rather than a flat SourceLink list --
/// DAF needs the full candidate set for every surface up front, since it is
/// reused across the whole annealing loop rather than discovered live during
/// navigation. Reuses @ref TrackFitterFunction::GeneralFitterOptions since
/// the propagation-level options are identical.
class DafFitterFunction {
 public:
  using TrackFitterResult = Acts::Result<TrackContainer::TrackProxy>;
  using GeneralFitterOptions = TrackFitterFunction::GeneralFitterOptions;

  virtual ~DafFitterFunction() = default;

  /// @param measurementGroups Competing candidates, grouped by surface
  /// @param surfaceSequence The proto track traversal order for the
  /// surfaces, used as the input to drive the DirectNavigator. By default
  /// it is used as-is (see DafFitterFunction.cpp for why geometric
  /// re-sorts hurt curlers); with @ref makeDafFitterFunction's
  /// useKfSequenceHandling a copy is run through the same sequence
  /// preparation as the KF (seed-path sort, material-surface injection,
  /// 1.5 m path limit) so both fitters traverse the identical sequence.
  /// @param finalCandidateState optional out-parameter; when non-null the
  /// fitter fills it with the final per-surface candidate weights/chi2 the
  /// returned track was built from (see DafFitterOptions::finalCandidateState)
  /// @param candidateTrace optional out-parameter; when non-null the fitter
  /// appends one entry per annealing pass, giving the whole weight history
  /// rather than just its endpoint (see DafFitterOptions::candidateTrace)
  /// @param fitStatus optional out-parameter; when non-null the fitter records
  /// how the fit ended (see DafFitterOptions::fitStatus)
  virtual TrackFitterResult operator()(
      const std::unordered_map<const Acts::Surface*, Acts::DafMeasurementGroup>&
          measurementGroups,
      const std::vector<const Acts::Surface*>& surfaceSequence,
      const TrackParameters&, const GeneralFitterOptions&,
      const MeasurementCalibratorAdapter&, TrackContainer&,
      Acts::DafCandidateMap* finalCandidateState = nullptr,
      Acts::DafCandidateTrace* candidateTrace = nullptr,
      Acts::DafFitStatus* fitStatus = nullptr) const = 0;
};

/// Makes a fitter function object for the Deterministic Annealing Filter
///
/// @param trackingGeometry the trackingGeometry for the propagator
/// @param magneticField the magnetic field for the propagator
/// @param multipleScattering bool
/// @param energyLoss bool
/// @param freeToBoundCorrection bool
/// @param annealingTemperatures the annealing temperature schedule, from hottest to coldest
/// @param annealingCutOff the insensitivity threshold for the calculated weight at cutoff
/// @param iterationsPerTemperature number of forward+smooth passes run at each temperature
/// @param maxIterations hard cap on the total number of forward+smooth passes
/// @param useDirectNavigator whether to use the DirectNavigator over the candidate group surfaces
/// @param useKfSequenceHandling prepare the DirectNavigator sequence exactly
/// like the KF fitter function does (straight-line-path sort from the seed,
/// material-surface injection, 1.5 m path limit) instead of trusting the
/// proto-track order. Needed for KF-equivalence comparisons and for material
/// treatment inside the fit; off by default because the proto-track order is
/// the safer choice for curlers and non-monotonic sequences.
/// @param logger a logger instance
std::shared_ptr<DafFitterFunction> makeDafFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    bool multipleScattering = true, bool energyLoss = true,
    Acts::FreeToBoundCorrection freeToBoundCorrection =
        Acts::FreeToBoundCorrection(),
    std::vector<double> annealingTemperatures = {64., 16., 4., 2., 1.5, 1.},
    double annealingCutOff = 9., std::size_t iterationsPerTemperature = 1,
    std::size_t maxIterations = 100, bool useDirectNavigator = true,
    bool useKfSequenceHandling = false,
    double strawInitialWeightRadiusFactor = 0.,
    bool useMomentMatchedCombiner = false,
    double reseedCovarianceScaling = 1.,
    Acts::detail::DafChiSquareDenominator chiSquareDenominator =
        Acts::detail::DafChiSquareDenominator::Full,
    bool annealingCutOffPerCandidate = false,
    bool referenceLinearization = false,
    bool referenceWeightCalibration = true,
    double weightConvergenceTolerance = 0.,
    std::size_t minAnnealingIterations = 1,
    double pValueConvergenceTolerance = 0., double pValueMinimum = 0.,
    std::size_t pValueMinIterations = 5,
    bool reseedResetOffDiagonals = false,
    double reseedVarianceCeiling = 0.,
    double reseedAngleVarianceCeiling = 0.,
    bool useTwoFilterSmoothing = false,
    double twoFilterBlowUpFactor = 1000.,
    bool referencePiecewise = false, bool referenceAnchorSmoothed = false,
    bool referencePiecewiseSelfCheck = false,
    bool blowUpInitialSeed = false,
    const Acts::Logger& logger = *Acts::getDefaultLogger(
        "Daf", Acts::Logging::INFO));

}  // namespace ActsExamples

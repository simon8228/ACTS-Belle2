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
  /// surfaces, used as the input to drive the DirectNavigator. When
  /// @ref makeDafFitterFunction's useDirectNavigator is set, the
  /// implementation re-sorts a copy of this by increasing geometryId layer
  /// (inner->outer) before propagating, mirroring
  /// KalmanFitterFunctionImpl -- proto-track order alone was found
  /// unreliable often enough (pattern recognition does not guarantee true
  /// spatial traversal order) to cause DirectNavigator to exhaust its
  /// sequence mid-track and fall back to the propagator's step-count
  /// limit. Note the layer sort has its own known blind spot: it isn't
  /// radius-monotonic for low-momentum tracks that curl back radially in
  /// the field, so it is not a full fix either.
  /// @param finalCandidateState optional out-parameter; when non-null the
  /// fitter fills it with the final per-surface candidate weights/chi2 the
  /// returned track was built from (see DafFitterOptions::finalCandidateState)
  virtual TrackFitterResult operator()(
      const std::unordered_map<const Acts::Surface*, Acts::DafMeasurementGroup>&
          measurementGroups,
      const std::vector<const Acts::Surface*>& surfaceSequence,
      const TrackParameters&, const GeneralFitterOptions&,
      const MeasurementCalibratorAdapter&, TrackContainer&,
      Acts::DafCandidateMap* finalCandidateState = nullptr) const = 0;
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
    const Acts::Logger& logger = *Acts::getDefaultLogger(
        "Daf", Acts::Logging::INFO));

}  // namespace ActsExamples

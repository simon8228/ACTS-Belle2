#pragma once

#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"

namespace ActsExamples {

class DriftChamberCalibrator : public MeasurementCalibrator {
public:
  /// The x-t re-evaluation reads its per-hit linearisation out of the cluster
  /// container (see calibrate()), so the fitting algorithm must be given one.
  /// Without this it silently receives a nullptr and the correction never runs.
  bool needsClusters() const override { return true; }

  void calibrate(
      const MeasurementContainer& measurements,
      const ClusterContainer* clusters, 
      const Acts::GeometryContext& gctx,
      const Acts::CalibrationContext& cctx, 
      const Acts::SourceLink& sourceLink,
      Acts::VectorMultiTrajectory::TrackStateProxy& trackState) const override;
};

} // namespace ActsExamples
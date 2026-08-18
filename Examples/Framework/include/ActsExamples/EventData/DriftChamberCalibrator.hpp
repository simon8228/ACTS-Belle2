#pragma once

#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"

namespace ActsExamples {

/// How the left/right sign of a drift radius is chosen once the x-t correction
/// has been applied. The two fitters need opposite behaviour, so this is not a
/// tuning knob -- picking the wrong one silently destroys the fit.
enum class DriftSignPolicy {
  /// Take the sign from the predicted state. A Kalman filter is handed ONE
  /// unsigned drift circle per wire and has no other information, so it must
  /// commit to the side its prediction favours.
  FromPrediction,
  /// Keep the sign the export wrote. The DAF resolves the ambiguity by carrying
  /// both mirror candidates (+mR and -mL) on the same surface and annealing
  /// between them. Re-signing both from the same prediction collapses them onto
  /// one value, which leaves the annealing nothing to weigh -- the measured
  /// failure mode is 72.6% of hits ending at weight < 0.1.
  Preserve,
};

class DriftChamberCalibrator : public MeasurementCalibrator {
public:
  /// @param signPolicy see DriftSignPolicy. Defaults to the Kalman behaviour so
  ///        that existing KF call sites are unchanged.
  explicit DriftChamberCalibrator(
      DriftSignPolicy signPolicy = DriftSignPolicy::FromPrediction)
      : m_signPolicy(signPolicy) {}

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

private:
  DriftSignPolicy m_signPolicy;
};

} // namespace ActsExamples
// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/MeasurementCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "Acts/EventData/SourceLink.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/DriftChamberCalibrator.hpp"
#include <cassert>

namespace ActsExamples {

void PassThroughCalibrator::calibrate(
    const MeasurementContainer& measurements,
    const ClusterContainer* /*clusters*/, const Acts::GeometryContext& /*gctx*/,
    const Acts::CalibrationContext& /*cctx*/,
    const Acts::SourceLink& sourceLink,
    Acts::VectorMultiTrajectory::TrackStateProxy& trackState) const {
  trackState.setUncalibratedSourceLink(Acts::SourceLink{sourceLink});
  const IndexSourceLink& idxSourceLink = sourceLink.get<IndexSourceLink>();

  assert((idxSourceLink.index() < measurements.size()) &&
         "Source link index is outside the container bounds");

  const ConstVariableBoundMeasurementProxy measurement =
      measurements.getMeasurement(idxSourceLink.index());

  Acts::visit_measurement(measurement.size(), [&](auto N) -> void {
    constexpr std::size_t kMeasurementSize = decltype(N)::value;
    const ConstFixedBoundMeasurementProxy<kMeasurementSize> fixedMeasurement =
        static_cast<ConstFixedBoundMeasurementProxy<kMeasurementSize>>(
            measurement);

    trackState.allocateCalibrated(fixedMeasurement.parameters().eval(),
                                  fixedMeasurement.covariance().eval());
    trackState.setProjectorSubspaceIndices(fixedMeasurement.subspaceIndices());
  });
}

MeasurementCalibratorAdapter::MeasurementCalibratorAdapter(
    const MeasurementCalibrator& calibrator,
    const MeasurementContainer& measurements, const ClusterContainer* clusters)
    : m_calibrator{calibrator},
      m_measurements{measurements},
      m_clusters{clusters} {}

void MeasurementCalibratorAdapter::calibrate(
    const Acts::GeometryContext& gctx, const Acts::CalibrationContext& cctx,
    const Acts::SourceLink& sourceLink,
    Acts::VectorMultiTrajectory::TrackStateProxy trackState) const {
  return m_calibrator.calibrate(m_measurements, m_clusters, gctx, cctx,
                                sourceLink, trackState);
}
void ActsExamples::DriftChamberCalibrator::calibrate(
    const MeasurementContainer& measurements,
    const ClusterContainer* clusters, 
    const Acts::GeometryContext& gctx,
    const Acts::CalibrationContext& cctx, 
    const Acts::SourceLink& sourceLink,
    Acts::VectorMultiTrajectory::TrackStateProxy& trackState) const {

    trackState.setUncalibratedSourceLink(Acts::SourceLink{sourceLink});
    const IndexSourceLink& idxSourceLink = sourceLink.get<IndexSourceLink>();

    auto rawMeasurement = measurements.at(idxSourceLink.index());

    Eigen::VectorXd params = rawMeasurement.parameters();
    Eigen::MatrixXd cov = rawMeasurement.covariance();
    
    auto indices = rawMeasurement.subspaceIndexVector(); 
    
    if (rawMeasurement.size() == 1 &&
        trackState.referenceSurface().type() == Acts::Surface::Straw) {
        if (indices[0] == Acts::eBoundLoc0 && trackState.hasPredicted()) {

            double absDrift = std::abs(params[0]);

            // ---- re-evaluate the x-t relation at the CURRENT track state ----
            // The drift length in the input file was evaluated once, before the
            // fit, at the *seed's* incidence angles. Those angles are wrong by
            // however wrong the seed was, and the x-t relation depends on them.
            // The extraction therefore also ships a linearisation of the
            // relation about the seed angles (cells.csv -> ClusterContainer,
            // channel0 is the key):
            //   0 = alpha0   1 = theta0
            //   2 = dr/dalpha    3 = dr/dtheta      [mm/rad]
            //   4 = dvar/dalpha  5 = dvar/dtheta    [mm^2/rad]
            // so the fit can correct both the drift length and its variance
            // against its own, improving estimate of the incidence angles.
            if (clusters != nullptr &&
                idxSourceLink.index() < clusters->size()) {
                double a0 = std::numeric_limits<double>::quiet_NaN();
                double t0 = std::numeric_limits<double>::quiet_NaN();
                double drda = 0., drdt = 0., dvda = 0., dvdt = 0.;
                for (const auto& ch : (*clusters)[idxSourceLink.index()].channels) {
                    switch (ch.bin[0]) {
                        case 0: a0   = ch.activation; break;
                        case 1: t0   = ch.activation; break;
                        case 2: drda = ch.activation; break;
                        case 3: drdt = ch.activation; break;
                        case 4: dvda = ch.activation; break;
                        case 5: dvdt = ch.activation; break;
                        default: break;
                    }
                }

                if (std::isfinite(a0) && std::isfinite(t0)) {
                    // alpha is defined exactly as in the extraction: the angle
                    // between the wire's transverse position and the track's
                    // transverse momentum. theta is the polar angle. Both come
                    // from the predicted state, which improves every iteration.
                    const Acts::Vector3 wire =
                        trackState.referenceSurface().center(gctx);
                    const double phiP = trackState.predicted()[Acts::eBoundPhi];
                    const double thetaP =
                        trackState.predicted()[Acts::eBoundTheta];
                    const double px = std::sin(thetaP) * std::cos(phiP);
                    const double py = std::sin(thetaP) * std::sin(phiP);
                    const double alphaP =
                        std::atan2(wire.x() * py - wire.y() * px,
                                   wire.x() * px + wire.y() * py);

                    double dAlpha = alphaP - a0;
                    // shortest way round: the linearisation is only valid for a
                    // small step, so a 2*pi wrap must not be taken literally.
                    while (dAlpha > M_PI) { dAlpha -= 2. * M_PI; }
                    while (dAlpha < -M_PI) { dAlpha += 2. * M_PI; }
                    const double dTheta = thetaP - t0;

                    static const bool applyLen = [] {
                        const char* e = std::getenv("ACTS_XT_LEN");
                        return e == nullptr || std::atoi(e) != 0;
                    }();
                    const double corrected =
                        applyLen ? absDrift + drda * dAlpha + drdt * dTheta
                                 : absDrift;
                    // Only accept a physically sensible correction; a wild
                    // predicted angle must not turn into a negative radius.
                    if (corrected > 0.) {
                        absDrift = corrected;
                    }

                    // The variance derivatives are large next to the variance
                    // itself (dvar/dalpha ~ -0.6 mm^2/rad vs var ~ 0.026 mm^2),
                    // so a modest angle change can drive the linearisation to
                    // zero or negative. Clamp to a factor of two either way, and
                    // allow it to be switched off entirely for A/B testing.
                    static const bool applyVar = [] {
                        const char* e = std::getenv("ACTS_XT_VAR");
                        return e == nullptr || std::atoi(e) != 0;
                    }();
                    if (applyVar) {
                        const double var =
                            cov(0, 0) + dvda * dAlpha + dvdt * dTheta;
                        cov(0, 0) = std::clamp(var, 0.5 * cov(0, 0),
                                               2.0 * cov(0, 0));
                    }
                }
            }

            // ---- left/right: sign the drift by the predicted side ----
            double predictedLoc0 = trackState.predicted()[Acts::eBoundLoc0];
            params[0] = (predictedLoc0 < 0.0) ? -absDrift : absDrift;
        }
    }

    trackState.allocateCalibrated(rawMeasurement.size());
    
    trackState.setProjectorSubspaceIndices(indices);
    

    trackState.effectiveCalibrated() = params;
    trackState.effectiveCalibratedCovariance() = cov;
}
}  // namespace ActsExamples

// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/MeasurementCalibration.hpp"

#include <atomic>
#include <memory>

namespace ActsExamples {

/// Belle II CDC calibrator that re-evaluates the x-t relation at the fit's own
/// track state, as GenFit's CDCRecoHit does on every pass.
///
/// The input drift lengths were evaluated once, before the fit, at the seed's
/// incidence angles. This calibrator keeps the side-independent drift time the
/// exporter shipped and redoes only the inversion
///   drift length = x-t(drift time, layer, side, alpha, theta)
/// plus the matching resolution, with alpha and theta taken from the state it
/// is handed (the predicted state while filtering, the reference state while
/// weighting). The x-t relation is the standalone port of basf2's
/// RealisticTDCCountTranslator (cdc-standalone), with one data set compiled in.
///
/// Every candidate keeps its own side: a DAF surface carries the left and the
/// right hypothesis as separate measurements, and each is re-evaluated with its
/// own, side-dependent x-t relation. Nothing is re-signed from the predicted
/// state.
///
/// Inputs per measurement, from cells.csv (the cluster container), keyed by
/// channel bin[0]:
///   10 drift time [ns]   11 side (basf2 leftRight: 0 left, 1 right)
///   12 sag shift [mm]    13 alpha [rad]   14 theta [rad] (export angles)
///   15-17 seed track point x, y, z [cm]   18 seed momentum phi [rad]
///   19 TOF the drift time is net of [ns]   20 the track's seed time (ACTS units)
/// A measurement without these inputs passes through as exported.
///
/// useTrackTime: the drift time is net of the flight time the exporter
/// estimated from the seed (key 19). With this set, that estimate is swapped
/// for the fit's own: the predicted time minus the seed time (key 20), i.e. the
/// time ACTS integrated along the track with the pion hypothesis -- what
/// CDCRecoHit passes as the track time. The exporter's guard is kept: a flight
/// time that is negative or not below the TOF-free drift time is not
/// subtracted at all.
///
/// A candidate whose drift time inverts to a non-positive length (a negative
/// drift time) passes through as exported, and then
/// - flagNonPositive: it is flagged as an outlier, and the DAF starts the hit
///   with weight 0 on both candidates on its first pass -- exactly GenFit's
///   CDCRecoHit (mL < 0 || mR < 0), whose zero weight is replaced by the DAF's
///   own from the second pass on;
/// - dropNonPositive: its variance is made so large that it carries no
///   information on every pass -- stronger than GenFit. (A calibrator cannot
///   set a DAF weight, so the weight plots still show such a hit.)
///
/// The wire sag: ACTS's straw is the straight c_Base wire, the real wire sags.
/// - recomputeSag = false: the shift the exporter folded into local0 (key 12,
///   evaluated once at the seed) is added back unchanged.
/// - recomputeSag = true: the shift is recomputed at every call, exactly as the
///   exporter does it but at the state handed in: the track point is projected
///   onto the tangent to the sagged wire at the track's z (cdc-standalone's port
///   of CDCGeometryPar::getWireSagEffect), and the offset from the straight
///   wire is projected on the drift direction w x p. alpha is then taken at that
///   point on the sagged wire. Needs a data set written with --wire-sag.
///
/// ACTS_XT_CHECK=1 additionally recomputes every candidate at the EXPORT
/// angles, and its sag shift at the export point, and counts disagreements with
/// the exported values -- a check that the data set compiled in matches the
/// conditions the sample was exported with.
class CdcXtCalibrator : public MeasurementCalibrator {
 public:
  struct Config {
    bool recomputeSag = false;
    bool dropNonPositive = false;
    bool flagNonPositive = false;
    bool useTrackTime = false;
  };

  explicit CdcXtCalibrator(const Config& cfg);
  ~CdcXtCalibrator() override;

  bool needsClusters() const override { return true; }

  void calibrate(
      const MeasurementContainer& measurements,
      const ClusterContainer* clusters, const Acts::GeometryContext& gctx,
      const Acts::CalibrationContext& cctx, const Acts::SourceLink& sourceLink,
      Acts::VectorMultiTrajectory::TrackStateProxy& trackState) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  void printSummary() const;

  Config m_cfg;

  // Telemetry, printed when the calibrator is destroyed.
  mutable std::atomic<long> m_nCalls{0};
  mutable std::atomic<long> m_nReevaluated{0};
  mutable std::atomic<long> m_nNoInputs{0};
  mutable std::atomic<long> m_nNonPositive{0};
  mutable std::atomic<long> m_nChecked{0};
  mutable std::atomic<long> m_nCheckFailed{0};
  mutable std::atomic<long> m_nSagChecked{0};
  mutable std::atomic<long> m_nSagCheckFailed{0};
  mutable std::atomic<long> m_nTof{0};
  mutable std::atomic<long> m_nTofGuard{0};
  mutable std::atomic<long> m_nTofMoved{0};
};

}  // namespace ActsExamples

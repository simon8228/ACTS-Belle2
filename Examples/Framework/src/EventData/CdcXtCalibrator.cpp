// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/CdcXtCalibrator.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef ACTS_BELLE2_HAVE_CDC_STANDALONE
#include <cdc_standalone/RealisticTDCCountTranslator.h>
#include <cdc_standalone/WireSag.h>

// The data set is chosen at configure time (ACTS_BELLE2_CDC_DATASET), e.g.
// exp1003run0 -> <exp1003run0/parameters.h>, cdc_standalone::data_exp1003run0.
#define ACTS_CDC_STR2(x) #x
#define ACTS_CDC_STR(x) ACTS_CDC_STR2(x)
#define ACTS_CDC_CAT2(a, b) a##b
#define ACTS_CDC_CAT(a, b) ACTS_CDC_CAT2(a, b)
#include ACTS_CDC_STR(ACTS_BELLE2_CDC_DATASET/parameters.h)
#endif

namespace ActsExamples {

namespace {
// cells.csv keys, see finding_basf2_cdc_daf.py EXPORT_XT_INPUTS
constexpr unsigned int kDriftTime = 10;
constexpr unsigned int kSide = 11;
constexpr unsigned int kSagShift = 12;
constexpr unsigned int kAlpha = 13;
constexpr unsigned int kTheta = 14;
constexpr unsigned int kSeedX = 15;
constexpr unsigned int kSeedY = 16;
constexpr unsigned int kSeedZ = 17;
constexpr unsigned int kSeedPhi = 18;
constexpr unsigned int kTof = 19;
constexpr unsigned int kSeedTime = 20;

// The cells are read as float32: agreement with the exported values is only
// good to ~1e-6 mm.
constexpr double kCheckToleranceMm = 1e-3;

// The exporter's MAX_SAG_SHIFT_CM: past it the parabola is running away, and
// the hit is left on the real wire (no shift) rather than moved by a wrong
// amount.
constexpr double kMaxSagShiftCm = 0.1;

// dropNonPositive: a variance (mm^2) at which a measurement carries no
// information -- a 10 m sigma against ~0.2 mm hits.
constexpr double kNoInformationVar = 1e8;
}  // namespace

#ifdef ACTS_BELLE2_HAVE_CDC_STANDALONE
struct CdcXtCalibrator::Impl {
  const cdc_standalone::DriftParameters& par =
      cdc_standalone::ACTS_CDC_CAT(data_, ACTS_BELLE2_CDC_DATASET)::parameters();
  // Simulation: selects the MC fudge factor of the resolution, as basf2 does
  // for a sample with Environment::isMC().
  cdc_standalone::RealisticTDCCountTranslator translator{
      par, cdc_standalone::TranslatorConfig{false, false, true}};
  bool check = [] {
    const char* e = std::getenv("ACTS_XT_CHECK");
    return e != nullptr && std::atoi(e) != 0;
  }();

  /// Signed local0 [mm] and its variance [mm^2] for one candidate; false when
  /// the drift time does not invert to a positive length.
  bool evaluate(double driftTime, unsigned short iCLayer, unsigned short iWire,
                int side, double sagShiftMm, double alpha, double theta,
                double& local0, double& var) const {
    const double lengthCm = cdc_standalone::xt::driftLength(
        par, driftTime, iCLayer, side, alpha, theta);
    if (!std::isfinite(lengthCm) || lengthCm <= 0.) {
      return false;
    }
    local0 = (side == 1 ? 1. : -1.) * lengthCm * 10. + sagShiftMm;
    // cm^2 -> mm^2; z is unused by the resolution, as in basf2
    var = translator.getDriftLengthResolution(
              lengthCm,
              cdc_standalone::WireID::fromContinuousLayer(iCLayer, iWire),
              side == 1, 0., alpha, theta) *
          100.;
    return true;
  }

  /// The exporter's USE_WIRE_SAG_CORRECTION at a given track point and
  /// direction (cm, unit vector): the signed shift [cm] of the real, sagged
  /// wire from the straight c_Base wire, projected on w x p, and the point on
  /// the sagged wire nearest the track point. False if it is not defined.
  bool sagShift(unsigned short iCLayer, unsigned short iWire,
                const Acts::Vector3& point, const Acts::Vector3& dir,
                double& shiftCm, Acts::Vector3& crossing) const {
    const unsigned index = iCLayer * cdc_standalone::c_maxNDriftCells + iWire;
    const Acts::Vector3 bwd(par.wireBaseBwd[3 * index],
                            par.wireBaseBwd[3 * index + 1],
                            par.wireBaseBwd[3 * index + 2]);
    const Acts::Vector3 fwd(par.wireBaseFwd[3 * index],
                            par.wireBaseFwd[3 * index + 1],
                            par.wireBaseFwd[3 * index + 2]);

    // getWireSagEffect is a parabola with no range check: clamp z to the
    // wire, as the exporter does.
    const double z = std::clamp(point.z(), std::min(bwd.z(), fwd.z()),
                                std::max(bwd.z(), fwd.z()));
    double sagBwd[3], sagFwd[3];
    cdc_standalone::wiresag::wireEndsAt(
        par, cdc_standalone::wiresag::WireSet::Base, iCLayer, iWire, z, sagBwd,
        sagFwd);
    const Acts::Vector3 tb(sagBwd[0], sagBwd[1], sagBwd[2]);
    const Acts::Vector3 tf(sagFwd[0], sagFwd[1], sagFwd[2]);
    const double tnorm = (tf - tb).norm();
    const double nnorm = (fwd - bwd).norm();
    if (!(tnorm > 0.) || !(nnorm > 0.)) {
      return false;
    }
    const Acts::Vector3 tdir = (tf - tb) / tnorm;
    crossing = tb + (point - tb).dot(tdir) * tdir;

    const Acts::Vector3 ndir = (fwd - bwd) / nnorm;
    const Acts::Vector3 straight = bwd + (point - bwd).dot(ndir) * ndir;
    const Acts::Vector3 driftDir = ndir.cross(dir);
    const double cnorm = driftDir.norm();
    if (!(cnorm > 0.)) {
      return false;
    }
    shiftCm = (crossing - straight).dot(driftDir) / cnorm;
    if (!std::isfinite(shiftCm)) {
      return false;
    }
    if (std::abs(shiftCm) > kMaxSagShiftCm) {
      shiftCm = 0.;
    }
    return true;
  }
};
#else
struct CdcXtCalibrator::Impl {};
#endif

CdcXtCalibrator::CdcXtCalibrator(const Config& cfg) : m_cfg(cfg) {
#ifdef ACTS_BELLE2_HAVE_CDC_STANDALONE
  m_impl = std::make_unique<Impl>();
  if (m_cfg.recomputeSag && !m_impl->par.hasWireSag()) {
    throw std::runtime_error(
        "CdcXtCalibrator: recomputeSag needs a cdc-standalone data set "
        "written with --wire-sag");
  }
  std::fprintf(stderr,
               "CdcXtCalibrator: x-t relation from cdc-standalone data set "
               "'%s' (globaltag %s, experiment %d, run %d), wire sag %s, "
               "negative drift times %s, flight time %s\n",
               ACTS_CDC_STR(ACTS_BELLE2_CDC_DATASET), m_impl->par.globaltag,
               m_impl->par.experiment, m_impl->par.run,
               m_cfg.recomputeSag ? "recomputed per call" : "frozen at export",
               m_cfg.dropNonPositive
                   ? "carry no information"
                   : (m_cfg.flagNonPositive ? "start at weight 0 (GenFit)"
                                            : "kept as exported"),
               m_cfg.useTrackTime ? "from the fit's track time"
                                  : "frozen at export (seed)");
#else
  throw std::runtime_error(
      "CdcXtCalibrator: built without cdc-standalone; configure with "
      "-DACTS_BELLE2_CDC_STANDALONE_DIR=<checkout>");
#endif
}

CdcXtCalibrator::~CdcXtCalibrator() {
  printSummary();
}

void CdcXtCalibrator::printSummary() const {
  std::fprintf(stderr,
               "CdcXtCalibrator: %ld calibrations, %ld re-evaluated, %ld "
               "without x-t inputs, %ld non-positive (%s)\n",
               m_nCalls.load(), m_nReevaluated.load(), m_nNoInputs.load(),
               m_nNonPositive.load(),
               m_cfg.dropNonPositive
                   ? "no information"
                   : (m_cfg.flagNonPositive ? "first-pass weight 0"
                                            : "kept as exported"));
  if (m_nTof.load() > 0) {
    std::fprintf(stderr,
                 "CdcXtCalibrator: flight time from the fit on %ld candidates: "
                 "%ld differ from the seed's by more than 0.1 ns, %ld not "
                 "subtracted (guard)\n",
                 m_nTof.load(), m_nTofMoved.load(), m_nTofGuard.load());
  }
  if (m_nChecked.load() > 0) {
    std::fprintf(stderr,
                 "CdcXtCalibrator: ACTS_XT_CHECK %ld of %ld candidates differ "
                 "from the exported local0 by more than %g mm at the export "
                 "angles\n",
                 m_nCheckFailed.load(), m_nChecked.load(), kCheckToleranceMm);
  }
  if (m_nSagChecked.load() > 0) {
    std::fprintf(stderr,
                 "CdcXtCalibrator: ACTS_XT_CHECK %ld of %ld sag shifts differ "
                 "from the exported one by more than %g mm at the export "
                 "point\n",
                 m_nSagCheckFailed.load(), m_nSagChecked.load(),
                 kCheckToleranceMm);
  }
}

void CdcXtCalibrator::calibrate(
    const MeasurementContainer& measurements, const ClusterContainer* clusters,
    const Acts::GeometryContext& gctx, const Acts::CalibrationContext& /*cctx*/,
    const Acts::SourceLink& sourceLink,
    Acts::VectorMultiTrajectory::TrackStateProxy& trackState) const {
  trackState.setUncalibratedSourceLink(Acts::SourceLink{sourceLink});
  const IndexSourceLink& idxSourceLink = sourceLink.get<IndexSourceLink>();
  assert((idxSourceLink.index() < measurements.size()) &&
         "Source link index is outside the container bounds");
  const ConstVariableBoundMeasurementProxy measurement =
      measurements.getMeasurement(idxSourceLink.index());
  // Running summary: the fit script ends on the Sequencer's FPE exception,
  // so the destructor's summary is not guaranteed to be printed.
  if ((m_nCalls.fetch_add(1) + 1) % 1000000 == 0) {
    printSummary();
  }

  Acts::visit_measurement(measurement.size(), [&](auto N) -> void {
    constexpr std::size_t kMeasurementSize = decltype(N)::value;
    const ConstFixedBoundMeasurementProxy<kMeasurementSize> fixedMeasurement =
        static_cast<ConstFixedBoundMeasurementProxy<kMeasurementSize>>(
            measurement);
    auto parameters = fixedMeasurement.parameters().eval();
    auto covariance = fixedMeasurement.covariance().eval();

    if constexpr (kMeasurementSize == 1) {
      do {
        if (m_impl == nullptr || clusters == nullptr ||
            idxSourceLink.index() >= clusters->size() ||
            fixedMeasurement.subspaceIndices()[0] != Acts::eBoundLoc0 ||
            !trackState.hasPredicted() ||
            trackState.referenceSurface().type() != Acts::Surface::Straw) {
          ++m_nNoInputs;
          break;
        }
        const double nan = std::nan("");
        double driftTime = nan, side = nan, sagShift = 0.;
        double alpha0 = nan, theta0 = nan;
        double seedX = nan, seedY = nan, seedZ = nan, seedPhi = nan;
        double tofSeed = nan, seedTime = nan;
        for (const auto& ch : (*clusters)[idxSourceLink.index()].channels) {
          switch (ch.bin[0]) {
            case kDriftTime: driftTime = ch.activation; break;
            case kSide: side = ch.activation; break;
            case kSagShift: sagShift = ch.activation; break;
            case kAlpha: alpha0 = ch.activation; break;
            case kTheta: theta0 = ch.activation; break;
            case kSeedX: seedX = ch.activation; break;
            case kSeedY: seedY = ch.activation; break;
            case kSeedZ: seedZ = ch.activation; break;
            case kSeedPhi: seedPhi = ch.activation; break;
            case kTof: tofSeed = ch.activation; break;
            case kSeedTime: seedTime = ch.activation; break;
            default: break;
          }
        }
        if (!std::isfinite(driftTime) || !(side == 0. || side == 1.)) {
          ++m_nNoInputs;
          break;
        }
#ifdef ACTS_BELLE2_HAVE_CDC_STANDALONE
        // Wire numbering as the exporter packs it: layer = ICLayer + 1,
        // sensitive = wire in layer + 1.
        const Acts::GeometryIdentifier geoId = fixedMeasurement.geometryId();
        const auto iCLayer = static_cast<unsigned short>(geoId.layer() - 1);
        const auto iWire = static_cast<unsigned short>(geoId.sensitive() - 1);
        const int lr = static_cast<int>(side);

        const auto& predicted = trackState.predicted();
        const double theta = predicted[Acts::eBoundTheta];
        const Acts::Vector3 direction =
            Acts::makeDirectionFromPhiTheta(predicted[Acts::eBoundPhi], theta);

        // The point on the wire alpha is measured at, and the sag shift.
        // alpha is the exporter's and CDCRecoHit's: the angle between the
        // wire's transverse position and the track's transverse momentum.
        Acts::Vector3 wirePoint;
        double shiftMm = sagShift;
        bool haveWirePoint = false;
        if (m_cfg.recomputeSag) {
          // The track point: the predicted state on the straw, in cm.
          const Acts::Vector3 trackPoint =
              trackState.referenceSurface().localToGlobal(
                  gctx,
                  Acts::Vector2(predicted[Acts::eBoundLoc0],
                                predicted[Acts::eBoundLoc1]),
                  direction) /
              10.;
          double shiftCm = 0.;
          if (m_impl->sagShift(iCLayer, iWire, trackPoint, direction, shiftCm,
                               wirePoint)) {
            shiftMm = shiftCm * 10.;
            haveWirePoint = true;
          }
        }
        if (!haveWirePoint) {
          // The straight straw where the track passes it (loc0 = 0 at the
          // predicted loc1) rather than the straw centre, which for a stereo
          // wire is off by the stereo rotation.
          wirePoint = trackState.referenceSurface().localToGlobal(
              gctx, Acts::Vector2(0., predicted[Acts::eBoundLoc1]), direction);
        }
        const double alpha =
            std::atan2(wirePoint.x() * direction.y() -
                           wirePoint.y() * direction.x(),
                       wirePoint.x() * direction.x() +
                           wirePoint.y() * direction.y());

        if (m_impl->check && std::isfinite(alpha0) && std::isfinite(theta0)) {
          double local0Export = 0., varExport = 0.;
          if (m_impl->evaluate(driftTime, iCLayer, iWire, lr, sagShift, alpha0,
                               theta0, local0Export, varExport)) {
            ++m_nChecked;
            if (std::abs(local0Export - parameters(0)) > kCheckToleranceMm) {
              ++m_nCheckFailed;
            }
          }
          if (m_cfg.recomputeSag && std::isfinite(seedX) &&
              std::isfinite(seedPhi) && sagShift != 0.) {
            double shiftSeedCm = 0.;
            Acts::Vector3 unused;
            if (m_impl->sagShift(
                    iCLayer, iWire, Acts::Vector3(seedX, seedY, seedZ),
                    Acts::makeDirectionFromPhiTheta(seedPhi, theta0),
                    shiftSeedCm, unused)) {
              ++m_nSagChecked;
              if (std::abs(shiftSeedCm * 10. - sagShift) > kCheckToleranceMm) {
                ++m_nSagCheckFailed;
              }
            }
          }
        }

        // The flight time: the seed's, frozen in key 10, or the fit's own.
        double driftTimeUsed = driftTime;
        if (m_cfg.useTrackTime && std::isfinite(tofSeed) &&
            std::isfinite(seedTime)) {
          const double driftTimeNoTof = driftTime + tofSeed;
          double tof =
              (predicted[Acts::eBoundTime] - seedTime) / Acts::UnitConstants::ns;
          ++m_nTof;
          if (!std::isfinite(tof) || tof < 0. || tof >= driftTimeNoTof) {
            tof = 0.;
            ++m_nTofGuard;
          }
          if (std::abs(tof - tofSeed) > 0.1) {
            ++m_nTofMoved;
          }
          driftTimeUsed = driftTimeNoTof - tof;
        }

        double local0 = 0., var = 0.;
        if (!m_impl->evaluate(driftTimeUsed, iCLayer, iWire, lr, shiftMm,
                              alpha, theta, local0, var)) {
          ++m_nNonPositive;
          if (m_cfg.dropNonPositive) {
            covariance(0, 0) = kNoInformationVar;
          } else if (m_cfg.flagNonPositive) {
            trackState.typeFlags().setIsOutlier();
          }
          break;
        }
        parameters(0) = local0;
        if (std::isfinite(var) && var > 0.) {
          covariance(0, 0) = var;
        }
        ++m_nReevaluated;
#endif
      } while (false);
    }

    trackState.allocateCalibrated(parameters, covariance);
    trackState.setProjectorSubspaceIndices(fixedMeasurement.subspaceIndices());
  });
}

}  // namespace ActsExamples

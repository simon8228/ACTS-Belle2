// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/MeasurementHelpers.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/SubspaceHelpers.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackParameterHelpers.hpp"
#include "Acts/EventData/TrackStatePropMask.hpp"
#include "Acts/TrackFitting/DafFitterError.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/TrackFitting/detail/DafReferenceState.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "Acts/Utilities/TrackHelpers.hpp"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

namespace Acts::detail {

/// Kalman-filter a recorded DAF reference trajectory, without propagating.
///
/// This is the second half of GenFit's KalmanFitterRefTrack: the reference
/// pass (DafActor with recordReference set) has already produced, per surface,
/// the reference parameters, the transport Jacobian F, the process noise N,
/// the delta state c and the reference-calibrated candidates. All that is left
/// is the affine prediction
///
///     p_k|k-1 = F_k * p_k-1|k-1 + c_k
///     C_k|k-1 = F_k * C_k-1|k-1 * F_k^T + N_k
///
/// followed by the usual weighted combination and gain-matrix update. No
/// stepper, no navigator, no material lookup -- every hit was linearised on
/// one and the same trajectory, which is the entire point.
///
/// The filter starts at the first recorded state with the reference parameters
/// and the supplied start covariance, exactly as GenFit does ("Use reference
/// state and seed cov as start"): the deviation from the reference starts at
/// zero, while the covariance claims no information.
///
/// @tparam traj_t The trajectory type
/// @tparam track_container_t The track container type
///
/// @param reference The recorded reference trajectory, in propagation order
/// @param candidateStore Persistent store holding the calibrated candidates
///        that DafReferenceState::candidateIndices point into
/// @param startCovariance Covariance the filter starts with, i.e. GenFit's
///        blown-up seed covariance
/// @param candidateState Per-surface annealed weights for this pass
/// @param options The fitter options, for the extensions and contexts
/// @param trackContainer Track container to build the filtered track in
/// @param logger Logger for diagnostic output
///
/// @return The filtered track, ready to be smoothed
template <typename traj_t, typename track_container_t>
Result<typename track_container_t::TrackProxy> dafReferenceFilter(
    const DafReferenceTrajectory& reference, const traj_t& candidateStore,
    const BoundSquareMatrix& startCovariance,
    const DafCandidateMap& candidateState,
    const DafFitterOptions<traj_t>& options, track_container_t& trackContainer,
    const Logger& logger) {
  if (reference.empty()) {
    ACTS_DEBUG("Reference trajectory is empty");
    return DafFitterError::NoMeasurementFound;
  }

  auto& trajectory = trackContainer.trackStateContainer();

  std::size_t previousIndex = kTrackIndexInvalid;
  std::size_t lastMeasurementIndex = kTrackIndexInvalid;
  std::size_t measurementStates = 0;

  BoundVector previousFiltered = BoundVector::Zero();
  BoundSquareMatrix previousFilteredCovariance = BoundSquareMatrix::Zero();
  BoundVector previousReference = BoundVector::Zero();
  BoundVector lastDeviation = BoundVector::Zero();
  bool hasPrevious = false;

  // Per-step trace of exactly what the affine map consumes: x_ref, the
  // deviation it is applied to, F and Q. Everything here is cached by the
  // reference pass -- nothing in this loop propagates or touches geometry --
  // so this is the whole input to the filter, in one place.
  const bool traceSteps = logger.doPrint(Logging::DEBUG);

  // Flag thresholds. Each is a statement about when a quantity stops meaning
  // what the filter assumes it means, not a tuned cut.
  constexpr double kDriftLoc0Warn = 2.0;    // mm, linear expansion in position
  constexpr double kDriftQopWarn = 0.05;    // relative, 5% in q/p
  constexpr double kChi2Warn = 25.0;        // one candidate against the fit
  std::size_t warnDrift = 0, warnChi2 = 0, alertZeroNoise = 0, alertNotPD = 0;
  auto fmt = [](const BoundVector& v) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(4) << "loc0=" << v[eBoundLoc0]
       << " loc1=" << v[eBoundLoc1] << " phi=" << v[eBoundPhi]
       << " theta=" << v[eBoundTheta] << " qop=" << v[eBoundQOverP];
    return os.str();
  };
  auto fmtSqrtDiag = [](const BoundSquareMatrix& m) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(3);
    const char* names[] = {"loc0", "loc1", "phi", "theta", "qop"};
    for (int i = 0; i < 5; ++i) {
      os << (i ? " " : "") << names[i] << "="
         << std::sqrt(std::max(0., m(i, i)));
    }
    return os.str();
  };
  // How far F departs from the identity, and the largest deviation the linear
  // model was asked to transport. A deviation that stops being small is the
  // signal that linearising about this reference has stopped being valid.
  // loc1 is reported SEPARATELY and deliberately. On a straw it is the
  // position along the wire, which no measurement constrains (H is loc0 only),
  // so the POCA slides freely and loc1 dominates a plain max-over-components
  // by two orders of magnitude -- it swamped this metric on the first run and
  // made a converging fit look divergent. The measured components are what say
  // whether linearising about this reference is still valid.
  double worstDeviation = 0., worstAlongWire = 0.;
  std::size_t worstDeviationIndex = 0;
  std::size_t stepIndex = 0;

  // Covariance conditioning. Two places in this scheme meet a covariance whose
  // condition number is large BY CONSTRUCTION -- the bound parametrisation
  // mixes mm and rad, so dloc1/dtheta runs to hundreds of mm/rad on a metre-long
  // wire, and the re-seed deliberately imposes an eight-order spread (ceilings
  // 1e6 on loc0/loc1/qop against 0.01 rad^2 on phi/theta):
  //
  //   * the gain-matrix update uses the SHORT form C <- (I - K H) C
  //     (GainMatrixUpdaterImpl.hpp:69), which is algebraically right but
  //     preserves neither symmetry nor positive-definiteness, and runs ~50
  //     times per pass;
  //   * the RTS smoother inverts the PREDICTED covariance outright.
  //
  // Symmetry error and the smallest eigenvalue are what detect either going
  // wrong. Only the 5x5 spatial block is examined: eBoundTime is unconstrained
  // here, so including it would report an eigenvalue near zero on every step
  // and say nothing about the parameters the fit actually determines.
  // Symplectic check, done as a RUNNING PRODUCT rather than per step.
  //
  // det(F) for a SINGLE step is not an invariant and must not be read as one.
  // ACTS composes F_bound = J(free->bound at B) . T(A->B) . J(bound->free at A);
  // only the middle factor is symplectic. The outer two are parametrisation
  // changes at different surfaces and cancel only when both present the same
  // frame -- a CDC layer boundary is a cylinder in (r.phi, z), a straw is a
  // line surface in (drift, along-wire) whose axis is wire x direction. So a
  // layer<->straw step carries the RATIO of those projections and reads far
  // from 1 while being perfectly correct.
  //
  // Measured over 36 passes: per-step det ranges 0.75-1.33, consecutive steps
  // are anti-correlated (lag-1 r = -0.69), and the product over ~110 steps has
  // median 1.0000, range [0.903, 1.023] -- against [0.394, 2.5] for a random
  // walk with the same per-step spread. The product telescopes to the frame
  // ratio between the track's two endpoints, and THAT is bounded near 1.
  // straw->straw steps alone, where the frames match, give 0.9966.
  double logDetProduct = 0.;
  std::size_t nonPositiveDet = 0;
  double worstAsymmetry = 0., smallestEigenvalue = 0., worstCondition = 0.;
  std::size_t worstAsymmetryIndex = 0, smallestEigenvalueIndex = 0,
              worstConditionIndex = 0, indefiniteCount = 0;
  bool haveEigenvalues = false;
  auto conditioning = [](const BoundSquareMatrix& cov) {
    const SquareMatrix<5> block = cov.topLeftCorner<5, 5>();
    const double asymmetry =
        (block - block.transpose()).cwiseAbs().maxCoeff();
    Eigen::SelfAdjointEigenSolver<SquareMatrix<5>> solver;
    // Symmetrised first: the solver reads one triangle, so feeding it an
    // asymmetric matrix would silently hide the very defect being looked for.
    solver.compute(0.5 * (block + block.transpose()),
                   Eigen::EigenvaluesOnly);
    const auto& lambda = solver.eigenvalues();
    return std::make_tuple(asymmetry, lambda(0), lambda(4));
  };
  auto recordConditioning = [&](const BoundSquareMatrix& cov,
                                std::size_t index) {
    const auto [asymmetry, minEigen, maxEigen] = conditioning(cov);
    if (asymmetry > worstAsymmetry) {
      worstAsymmetry = asymmetry;
      worstAsymmetryIndex = index;
    }
    if (!haveEigenvalues || minEigen < smallestEigenvalue) {
      smallestEigenvalue = minEigen;
      smallestEigenvalueIndex = index;
      haveEigenvalues = true;
    }
    if (minEigen <= 0.) {
      ++indefiniteCount;
    } else if (maxEigen / minEigen > worstCondition) {
      worstCondition = maxEigen / minEigen;
      worstConditionIndex = index;
    }
    return std::make_tuple(asymmetry, minEigen, maxEigen);
  };

  for (const DafReferenceState& ref : reference) {
    const TrackStatePropMask mask =
        ref.hasMeasurement
            ? (TrackStatePropMask::Predicted | TrackStatePropMask::Filtered |
               TrackStatePropMask::Jacobian | TrackStatePropMask::Calibrated)
            : (TrackStatePropMask::Predicted | TrackStatePropMask::Jacobian);

    auto trackStateProxy = trajectory.makeTrackState(mask, previousIndex);
    trackStateProxy.setReferenceSurface(ref.surface->getSharedPtr());
    trackStateProxy.jacobian() = ref.jacobian;
    trackStateProxy.pathLength() = ref.pathLength;

    if (!hasPrevious) {
      // Start of the track: the estimate IS the reference, so the deviation is
      // zero by construction. The covariance is the start covariance TRANSPORTED
      // onto this surface -- J*C*J^T + N, with J and N recorded from the
      // propagation start. Using the start covariance unchanged would apply a
      // covariance defined on the seed surface (a perigee) to a straw surface,
      // where its loc0/loc1 block means something entirely different.
      trackStateProxy.predicted() = ref.parameters;
      trackStateProxy.predictedCovariance() =
          ref.jacobian * startCovariance * ref.jacobian.transpose() + ref.noise;
    } else {
      // p_k|k-1 = p_ref,k + F_k * (p_k-1|k-1 - p_ref,k-1)
      //
      // Algebraically identical to GenFit's p = F*p + c, since
      // c_k = p_ref,k - F_k*p_ref,k-1 -- but written in the deviation form
      // because ACTS bound parameters carry phi and theta, and the raw affine
      // form cannot survive the branch cut. If the filtered estimate and the
      // reference land on opposite sides of +-pi, their componentwise
      // difference is ~2*pi rather than ~0 and the prediction is destroyed.
      // GenFit never meets this: its 5D plane parameters (q/p, u', v', u, v)
      // have no periodic coordinate. subtract/addBoundParameters wrap both
      // angles into the symmetric range, which is exactly what the small
      // deviations this filter works with need.
      const BoundVector deviation =
          subtractBoundParameters(previousFiltered, previousReference);
      trackStateProxy.predicted() =
          addBoundParameters(ref.parameters, ref.jacobian * deviation);
      trackStateProxy.predictedCovariance() =
          ref.jacobian * previousFilteredCovariance * ref.jacobian.transpose() +
          ref.noise;

      BoundVector measured = deviation.cwiseAbs();
      measured[eBoundLoc1] = 0.;
      const double maxDeviation = measured.maxCoeff();
      if (maxDeviation > worstDeviation) {
        worstDeviation = maxDeviation;
        worstDeviationIndex = stepIndex;
      }
      worstAlongWire =
          std::max(worstAlongWire, std::abs(deviation[eBoundLoc1]));
      lastDeviation = deviation;
    }

    if (traceSteps) {
      const BoundVector predicted = trackStateProxy.predicted();
      const BoundSquareMatrix& covPred = trackStateProxy.predictedCovariance();
      const auto [asymP, minP, maxP] =
          recordConditioning(covPred, stepIndex);
      const double condP =
          minP > 0. ? maxP / minP : std::numeric_limits<double>::infinity();

      // NOT the annealing pass -- that is the "DAF pass k/N at T=..." banner,
      // and there are at most maxIterations of those. This index counts
      // SURFACES along the reference trajectory, of which a CDC track has
      // ~115: roughly 50 straws plus the material layers between them.
      ACTS_DEBUG("[Ref surface " << stepIndex << "/" << reference.size()
                 << "] " << ref.surface->geometryId()
                 << " | Type: "
                 << (ref.hasMeasurement ? "MEASUREMENT"
                                        : (ref.isHole ? "HOLE" : "MATERIAL"))
                 << " | s = " << ref.pathLength << " mm");

      // Linearisation distance -- how far the estimate has moved from the
      // point F and the calibration were evaluated at. loc1 is quoted
      // separately throughout because nothing measures it: on a straw it is
      // the position along the wire, so the POCA slides freely and a plain
      // norm over all components reports that slide and nothing else.
      const double driftLoc0 = std::abs(lastDeviation[eBoundLoc0]);
      const double refQop = ref.parameters[eBoundQOverP];
      const double relQop =
          std::abs(predicted[eBoundQOverP] - refQop) /
          std::max(std::abs(refQop), std::numeric_limits<double>::min());
      const bool drifted =
          hasPrevious && (driftLoc0 > kDriftLoc0Warn || relQop > kDriftQopWarn);
      warnDrift += static_cast<std::size_t>(drifted);
      ACTS_DEBUG("  State (pred) : loc0=" << predicted[eBoundLoc0]
                 << " loc1=" << predicted[eBoundLoc1]
                 << " qop=" << predicted[eBoundQOverP]
                 << "  (ref qop=" << refQop << ", d|qop|/qop="
                 << 100. * relQop << "%, dx_lin loc0=" << driftLoc0
                 << " mm, loc1=" << std::abs(lastDeviation[eBoundLoc1])
                 << " mm)");
      if (drifted) {
        ACTS_DEBUG("  [WARN] Linearization drift: the linear expansion of F "
                   "and H is being applied far from where they were evaluated");
      }

      // det(F) is a direct check on the stepper: the transport of a charged
      // particle through a static field is symplectic, so away from material
      // det(F) must stay at 1. A drift away from 1 is the integrator failing,
      // not the fit disagreeing with anything.
      const BoundMatrix fMinusIdentity = ref.jacobian - BoundMatrix::Identity();
      const double detF = ref.jacobian.determinant();
      if (detF > 0.) {
        logDetProduct += std::log(detF);
      } else {
        ++nonPositiveDet;
      }
      ACTS_DEBUG("  Prop Matrix  : max|F-I| = "
                 << fMinusIdentity.cwiseAbs().maxCoeff()
                 << ", running prod det(F) = " << std::exp(logDetProduct)
                 << ", diag = " << ref.jacobian(0, 0) << " "
                 << ref.jacobian(1, 1) << " " << ref.jacobian(2, 2) << " "
                 << ref.jacobian(3, 3) << " " << ref.jacobian(4, 4));

      ACTS_DEBUG("  Covariance   : cond(C_pred) = " << condP
                 << ", min_eigen = " << minP << ", tr = " << covPred.trace()
                 << ", |C-C^T| = " << asymP);
      if (minP <= 0.) {
        ++alertNotPD;
        ACTS_DEBUG("  [FATAL] Non-PD covariance: smallest eigenvalue "
                   << minP << " -- the update has lost positive-definiteness");
      }

      // Material and the noise it produced, over the SAME segment. Both are
      // accumulated between recorded surfaces and reset together, so they can
      // be compared directly -- which is what makes the alert below mean
      // something. A surface merely FLAGGED hasMaterial with a zero Q is
      // normal: its material lands in the next segment.
      if (ref.thicknessInX0 > 0. || ref.hasMaterial) {
        const double qTheta = ref.noise(eBoundTheta, eBoundTheta);
        const double qQop = ref.noise(eBoundQOverP, eBoundQOverP);
        ACTS_DEBUG("  Material     : x/X0 = " << ref.thicknessInX0
                   << ", dE = " << ref.energyLoss / UnitConstants::MeV
                   << " MeV   [segment ending here]");
        ACTS_DEBUG("  Proc Noise Q : Q(theta,theta) = " << qTheta
                   << " (MS), Q(qop,qop) = " << qQop << " (straggling)"
                   << ", sqrt(diag Q): " << fmtSqrtDiag(ref.noise));
        if (ref.thicknessInX0 > 0. && qTheta == 0. && qQop == 0.) {
          ++alertZeroNoise;
          ACTS_DEBUG("  [ALERT] Zero material noise: x/X0 = "
                     << ref.thicknessInX0
                     << " was crossed but Q is identically zero -- the noise "
                        "calculator produced nothing from a real slab");
        }
      }
    }

    auto typeFlags = trackStateProxy.typeFlags();
    typeFlags.setHasParameters();
    if (ref.hasMaterial) {
      typeFlags.setHasMaterial();
    }
    if (ref.isHole) {
      typeFlags.setIsHole();
    }

    if (ref.hasMeasurement) {
      const auto candStateIt = candidateState.find(ref.surface);
      if (candStateIt == candidateState.end()) {
        ACTS_DEBUG("No candidate state for surface "
                   << ref.surface->geometryId());
        return DafFitterError::NoMeasurementFound;
      }
      const DafCandidateState& candState = candStateIt->second;
      if (candState.weights.size() != ref.candidateIndices.size()) {
        ACTS_DEBUG("Weight/candidate count mismatch on surface "
                   << ref.surface->geometryId() << ": "
                   << candState.weights.size() << " vs "
                   << ref.candidateIndices.size());
        return DafFitterError::NoMeasurementFound;
      }

      std::vector<typename traj_t::ConstTrackStateProxy> constCandidates;
      constCandidates.reserve(ref.candidateIndices.size());
      for (std::size_t index : ref.candidateIndices) {
        constCandidates.push_back(candidateStore.getTrackState(index));
      }

      // Per candidate, BEFORE they are collapsed into one effective
      // measurement -- once combined the individual residuals are gone, and
      // they are what says whether a mirror pair is being separated or is
      // still sharing weight.
      if (traceSteps) {
        for (std::size_t i = 0; i < constCandidates.size(); ++i) {
          const auto& candidate = constCandidates[i];
          visit_measurement(candidate.calibratedSize(), [&](auto N) {
            constexpr std::size_t kSize = decltype(N)::value;
            const auto subspace = FixedBoundSubspaceHelper<kSize>(
                candidate.template projectorSubspaceIndices<kSize>());
            const auto H = subspace.projector();
            const Vector<kSize> residual =
                candidate.template calibrated<kSize>() -
                H * trackStateProxy.predicted();
            const SquareMatrix<kSize> S =
                H * trackStateProxy.predictedCovariance() * H.transpose() +
                candidate.template calibratedCovariance<kSize>();
            const double chi2 =
                (residual.transpose() * S.inverse() * residual).value();
            std::ostringstream pulls;
            pulls << std::fixed << std::setprecision(2);
            for (std::size_t d = 0; d < kSize; ++d) {
              pulls << (d ? ", " : "")
                    << residual[d] / std::sqrt(std::max(S(d, d), 1e-300));
            }
            warnChi2 += static_cast<std::size_t>(chi2 > kChi2Warn);
            ACTS_DEBUG("  Measurement  : cand " << i << "/"
                       << constCandidates.size() << "  res = ["
                       << residual.transpose() << "]  S_diag = ["
                       << S.diagonal().transpose() << "]  chi2_inc = " << chi2
                       << "  pull = [" << pulls.str() << "]  DAF_weight = "
                       << candState.weights[i]);
            if (chi2 > kChi2Warn) {
              ACTS_DEBUG("  [WARN] High chi2: candidate " << i << " sits "
                         << std::sqrt(chi2)
                         << " sigma from the prediction -- an outlier, or the "
                            "prediction has drifted");
            }
          });
        }
      }

      options.extensions.weightedCombiner(constCandidates, candState.weights,
                                          trackStateProxy);

      // Same convention as DafActor: the state resolves to the currently
      // favoured candidate, which becomes exact once the weights sharpen.
      std::size_t bestIdx = 0;
      for (std::size_t i = 1; i < candState.weights.size(); ++i) {
        if (candState.weights[i] > candState.weights[bestIdx]) {
          bestIdx = i;
        }
      }
      if (bestIdx < candState.candidates.size()) {
        trackStateProxy.setUncalibratedSourceLink(
            SourceLink(candState.candidates[bestIdx]));
      }

      auto updateRes =
          options.extensions.updater(options.geoContext, trackStateProxy,
                                     logger);
      if (!updateRes.ok()) {
        ACTS_DEBUG("Update step failed on surface "
                   << ref.surface->geometryId() << ": " << updateRes.error());
        return updateRes.error();
      }
      typeFlags.setIsMeasurement();

      if (traceSteps) {
        const BoundSquareMatrix& covFilt = trackStateProxy.filteredCovariance();
        const auto [asymF, minF, maxF] = recordConditioning(covFilt, stepIndex);
        // tr(C) before against after: the update must remove information, so
        // the trace has to fall. A trace that GROWS across an update is the
        // short-form C <- (I - K H) C having gone wrong.
        ACTS_DEBUG("  Post-update  : tr(C) "
                   << trackStateProxy.predictedCovariance().trace() << " -> "
                   << covFilt.trace() << ", cond = "
                   << (minF > 0. ? maxF / minF
                                 : std::numeric_limits<double>::infinity())
                   << ", min_eigen = " << minF
                   << ", chi2_inc = " << trackStateProxy.chi2());
        if (minF <= 0.) {
          ++alertNotPD;
          ACTS_DEBUG("  [FATAL] Non-PD covariance after update: smallest "
                     "eigenvalue " << minF);
        }
        if (covFilt.trace() > trackStateProxy.predictedCovariance().trace()) {
          ACTS_DEBUG("  [WARN] tr(C) grew across the update -- an update "
                     "cannot add information");
        }
      }

      lastMeasurementIndex = trackStateProxy.index();
      ++measurementStates;
    } else {
      trackStateProxy.shareFrom(trackStateProxy, TrackStatePropMask::Predicted,
                                TrackStatePropMask::Filtered);
    }

    previousIndex = trackStateProxy.index();
    previousFiltered = trackStateProxy.filtered();
    previousFilteredCovariance = trackStateProxy.filteredCovariance();
    previousReference = ref.parameters;
    hasPrevious = true;
    ++stepIndex;
  }

  ACTS_DEBUG("Reference filter: " << reference.size() << " step(s), "
             << measurementStates
             << " measurement(s); largest deviation the linear model "
                "transported, EXCLUDING the unmeasured along-wire coordinate: "
                "max |dx| = "
             << worstDeviation << " at step " << worstDeviationIndex
             << "  (max |dx_loc1| = " << worstAlongWire << " separately)");
  if (traceSteps) {
    const double detProduct = std::exp(logDetProduct);
    ACTS_DEBUG("Symplectic check: product of det(F) over "
               << reference.size() << " step(s) = " << detProduct
               << " (telescopes to the frame ratio between the track's "
                  "endpoints; per-step values are NOT invariants)");
    // Wide on purpose. The bound is what a genuinely non-symplectic stepper
    // would blow through -- a random walk with the per-step spread seen here
    // reaches [0.39, 2.5] over one track, so anything inside this range is
    // indistinguishable from correct transport plus frame changes.
    if (!(detProduct > 0.5 && detProduct < 2.0)) {
      ACTS_DEBUG("  [WARN] Non-symplectic transport: the product should "
                 "telescope to O(1); per-step frame ratios cannot accumulate "
                 "this far, so suspect the stepper or the bound<->free "
                 "Jacobians");
    }
    if (nonPositiveDet != 0) {
      ACTS_DEBUG("  [FATAL] " << nonPositiveDet
                 << " Jacobian(s) with det(F) <= 0 -- transport cannot "
                    "reverse orientation");
    }
  }
  if (warnDrift || warnChi2 || alertZeroNoise || alertNotPD) {
    ACTS_DEBUG("Reference filter flags: " << warnDrift
               << " x [WARN] linearization drift, " << warnChi2
               << " x [WARN] high chi2, " << alertZeroNoise
               << " x [ALERT] zero material noise, " << alertNotPD
               << " x [FATAL] non-PD covariance");
  }
  if (haveEigenvalues) {
    ACTS_DEBUG("Reference filter conditioning: worst |C-C^T| = "
               << worstAsymmetry << " at step " << worstAsymmetryIndex
               << "; smallest eigenvalue = " << smallestEigenvalue
               << " at step " << smallestEigenvalueIndex
               << "; worst finite cond(C) = " << worstCondition << " at step "
               << worstConditionIndex << "; non-positive-definite covariances: "
               << indefiniteCount);
  }

  if (measurementStates == 0) {
    ACTS_DEBUG("Reference filter found no measurement states");
    return DafFitterError::NoMeasurementFound;
  }

  auto track = trackContainer.makeTrack();
  track.tipIndex() = lastMeasurementIndex;
  calculateTrackQuantities(track);

  return track;
}

}  // namespace Acts::detail

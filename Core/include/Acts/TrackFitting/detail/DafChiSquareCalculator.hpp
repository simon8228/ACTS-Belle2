// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/EventData/MeasurementHelpers.hpp"
#include "Acts/EventData/SubspaceHelpers.hpp"

namespace Acts::detail {

/// Which covariance the DAF's per-candidate chi2 divides its residual by.
///
/// The residual itself is the same in every case, r = m - H*x, and so is the
/// form chi2 = r^T S^-1 r. Only S differs -- and since the annealed weight is
/// exp(-chi2 / 2T), a smaller S means a larger chi2 means a sharper decision
/// between competing candidates. On a CDC wire the competing candidates are the
/// two drift-circle mirrors, so this is directly how decisively the fitter
/// resolves left/right.
///
/// The three are ordered: Unbiased < MeasurementOnly < Full.
enum class DafChiSquareDenominator {
  /// S = H C H^T + R. Correct when the state is INDEPENDENT of the measurement
  /// -- a forward prediction, or a smoothed state with this surface's own
  /// measurement subtracted out (DafFitterOptions::useUnbiasedSmoothedState).
  /// Wrong, and too large, against a plain biased smoothed state.
  Full,
  /// S = R, the measurement covariance alone. What GenFit's DAF uses:
  /// DAF.cc calls getResidual(j, biased=true, onlyMeasurementErrors=true), and
  /// KalmanFitterInfo::getResidual then returns measurement->getCov() without
  /// adding H C H^T. Ignores the state uncertainty entirely, which happens to
  /// land closer to the truth than Full does for a biased state.
  MeasurementOnly,
  /// S = R - H C H^T, the statistically correct covariance of a residual taken
  /// against a BIASED smoothed state. That state already absorbed this very
  /// measurement, so it sits closer to the hit than an independent estimate
  /// would and the residual is correspondingly smaller; subtracting is what
  /// accounts for it. Can fail to be positive definite when the measurement
  /// dominates the prior, in which case this falls back to MeasurementOnly.
  Unbiased,
};

/// Default per-candidate chi2 calculator for the Deterministic Annealing
/// Filter.
///
/// Computes the pre-update chi2 of one calibrated candidate measurement against
/// the (shared) predicted state:
///   r = m - H * predicted
///   S = per @c kDenominator, see DafChiSquareDenominator
///   chi2 = r^T S^-1 r
///
/// This deliberately uses the predicted state, not the filtered/post-update
/// residual formula used internally by @ref GainMatrixUpdater -- annealing
/// weights must be known before the candidates are combined and the update
/// runs, so they cannot depend on an update that itself depends on the
/// weights. Note that the DAF's outer driver calls this with `predicted` set to
/// the previous pass's SMOOTHED state, which is why the denominator question
/// arises at all.
///
/// @tparam traj_t The trajectory type
/// @tparam kDenominator Which covariance to divide by
/// @param candidate The calibrated candidate track state (its predicted
/// component must already be set, e.g. shared from a sibling candidate)
template <typename traj_t,
          DafChiSquareDenominator kDenominator = DafChiSquareDenominator::Full>
double defaultDafChiSquareCalculator(
    typename traj_t::ConstTrackStateProxy candidate) {
  const std::size_t measdim = candidate.calibratedSize();

  double chi2 = 0.;
  visit_measurement(measdim, [&](auto N) {
    constexpr std::size_t kMeasurementSize = decltype(N)::value;
    using ProjectedVector = Vector<kMeasurementSize>;
    using ProjectedMatrix = SquareMatrix<kMeasurementSize>;

    const auto measured = candidate.template calibrated<kMeasurementSize>();
    const auto measuredCov =
        candidate.template calibratedCovariance<kMeasurementSize>();

    const auto validSubspaceIndices =
        candidate.template projectorSubspaceIndices<kMeasurementSize>();
    const FixedBoundSubspaceHelper<kMeasurementSize> subspaceHelper(
        validSubspaceIndices);
    const auto H = subspaceHelper.projector();

    const auto predicted = candidate.predicted();
    const auto predictedCovariance = candidate.predictedCovariance();

    const ProjectedVector residual = measured - H * predicted;
    const ProjectedMatrix stateCovariance =
        H * predictedCovariance * H.transpose();

    ProjectedMatrix denominator;
    if constexpr (kDenominator == DafChiSquareDenominator::Full) {
      denominator = stateCovariance + measuredCov;
    } else if constexpr (kDenominator ==
                         DafChiSquareDenominator::MeasurementOnly) {
      denominator = measuredCov;
    } else {
      denominator = measuredCov - stateCovariance;
      // The subtraction is exact in principle but ill-conditioned exactly where
      // the measurement dominates the prior, and a non-positive-definite
      // denominator would hand back a negative chi2 and a weight above 1.
      // Falling back to the measurement covariance keeps the value finite and
      // keeps the fallback ADJACENT to this variant rather than jumping past
      // Full -- the three are ordered, so the nearest safe choice is the next
      // one up. For the 1D drift measurements this matters for, the test below
      // is just "is the scalar still positive".
      const Eigen::LLT<ProjectedMatrix> llt(denominator);
      if (llt.info() != Eigen::Success) {
        denominator = measuredCov;
      }
    }

    chi2 = (residual.transpose() * denominator.inverse() * residual).value();
  });
  return chi2;
}

}  // namespace Acts::detail

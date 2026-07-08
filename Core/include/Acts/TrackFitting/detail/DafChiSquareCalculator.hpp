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

/// Default per-candidate chi2 calculator for the Deterministic Annealing
/// Filter.
///
/// Computes the standard pre-update chi2 of one calibrated candidate
/// measurement against the (shared) predicted state:
///   r = m - H * predicted
///   S = H * predictedCovariance * H^T + R
///   chi2 = r^T S^-1 r
///
/// This deliberately uses the predicted state, not the filtered/post-update
/// residual formula used internally by @ref GainMatrixUpdater -- annealing
/// weights must be known before the candidates are combined and the update
/// runs, so they cannot depend on an update that itself depends on the
/// weights.
///
/// @tparam traj_t The trajectory type
/// @param candidate The calibrated candidate track state (its predicted
/// component must already be set, e.g. shared from a sibling candidate)
template <typename traj_t>
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
    const ProjectedMatrix innovationCovariance =
        H * predictedCovariance * H.transpose() + measuredCov;

    chi2 =
        (residual.transpose() * innovationCovariance.inverse() * residual)
            .value();
  });
  return chi2;
}

}  // namespace Acts::detail

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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace Acts::detail {

/// Default weighted-measurement combiner for the Deterministic Annealing
/// Filter.
///
/// Combines N calibrated candidate measurements on the same surface, all
/// sharing the same projected subspace, into a single effective measurement
/// using the standard information-form weighted combination:
///   R_eff^-1 = sum_i w_i R_i^-1
///   m_eff    = R_eff * sum_i w_i R_i^-1 m_i
///
/// @note This assumes all candidates measure the same subspace, e.g. distinct
/// clusters on overlapping silicon modules. It is not appropriate for
/// measurements that are mirror-image reinterpretations of the same raw
/// measurement (e.g. drift-circle left/right ambiguity), which need a
/// detector-specific @c WeightedCombiner instead.
///
/// @tparam traj_t The trajectory type
/// @param candidates The calibrated candidate track states, one per hit
/// hypothesis, same order as @p weights
/// @param weights The annealed weight of each candidate, same order as
/// @p candidates
/// @param combined The track state to write the effective measurement into;
/// its predicted parameters must already be set
template <typename traj_t>
void defaultDafWeightedCombiner(
    const std::vector<typename traj_t::ConstTrackStateProxy>& candidates,
    const std::vector<double>& weights,
    typename traj_t::TrackStateProxy combined) {
  assert(candidates.size() == weights.size() &&
         "defaultDafWeightedCombiner requires one weight per candidate");
  assert(!candidates.empty() &&
         "defaultDafWeightedCombiner requires at least one candidate");

  const std::size_t measdim = candidates.front().calibratedSize();

  visit_measurement(measdim, [&](auto N) {
    constexpr std::size_t kMeasurementSize = decltype(N)::value;
    using ProjectedVector = Vector<kMeasurementSize>;
    using ProjectedMatrix = SquareMatrix<kMeasurementSize>;

    ProjectedMatrix infoSum = ProjectedMatrix::Zero();
    ProjectedVector infoWeightedMeanSum = ProjectedVector::Zero();

    // Floor each weight away from exact zero. Weights come from
    // AnnealingUtility::getWeight(), which can genuinely underflow to 0.0
    // in double precision for a candidate with a very large chi2 at low
    // temperature (observed in practice for outlier hits after several
    // annealing iterations). Without a floor, a single-candidate group
    // whose only candidate underflows this way makes infoSum the exact
    // zero matrix, and its inverse is +-Inf/NaN, corrupting everything
    // downstream. Flooring keeps such a candidate's effective contribution
    // negligible while keeping the combination numerically well-defined.
    constexpr double kMinWeight = 1e-6;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      assert(candidate.calibratedSize() == measdim &&
             "defaultDafWeightedCombiner requires all candidates to share "
             "the same measurement subspace");

      const auto candMeas = candidate.template calibrated<kMeasurementSize>();
      const auto candCov =
          candidate.template calibratedCovariance<kMeasurementSize>();

      const double weight = std::max(weights[i], kMinWeight);
      const ProjectedMatrix candInfo = candCov.inverse();
      infoSum += weight * candInfo;
      infoWeightedMeanSum += weight * candInfo * candMeas;
    }

    const ProjectedMatrix combinedCov = infoSum.inverse();
    const ProjectedVector combinedMeas = combinedCov * infoWeightedMeanSum;

    combined.allocateCalibrated(combinedMeas, combinedCov);
    combined.setProjectorSubspaceIndices(
        candidates.front()
            .template projectorSubspaceIndices<kMeasurementSize>());
  });
}

}  // namespace Acts::detail

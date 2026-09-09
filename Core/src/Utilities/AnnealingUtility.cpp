// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/AnnealingUtility.hpp"

#include "Acts/Utilities/AlgebraHelpers.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace {
/// @brief Gaussian-like function for weight calculation
/// Note: Factor 2 in denominator is included in inverse temperature
///
/// @param chi2 Chi^2 value
/// @param invTemp Denominator 1/(2 * temperature)
///
/// @return exp(-chi2 * invTemp)
double computeAnnealingWeight(double chi2, double invTemp) {
  return Acts::safeExp(-chi2 * invTemp);
}
}  // namespace

Acts::AnnealingUtility::Config::Config() = default;

void Acts::AnnealingUtility::anneal(State& state) const {
  if (state.currentTemperatureIndex < m_cfg.setOfTemperatures.size() - 1) {
    ++state.currentTemperatureIndex;
  } else {
    state.equilibriumReached = true;
  }
}

double Acts::AnnealingUtility::getWeight(
    State& state, double chi2, const std::vector<double>& allChi2) const {
  unsigned int idx = state.currentTemperatureIndex;
  // Calculate 1/denominator in exp function already here
  const double currentInvTemp = 1. / (2. * m_cfg.setOfTemperatures[idx]);

  double num = computeAnnealingWeight(chi2, currentInvTemp);

  // One cut-off term per call, or one per competing candidate -- see
  // Config::cutOffPerCandidate. The latter is GenFit's convention.
  double denom = m_cfg.cutOffPerCandidate
                     ? static_cast<double>(allChi2.size()) *
                           m_gaussCutTempVec[idx]
                     : m_gaussCutTempVec[idx];

  for (double val : allChi2) {
    denom += computeAnnealingWeight(val, currentInvTemp);
  }

  return num / denom;
}

namespace {

/// log(DBL_MIN). Below it std::exp underflows to a subnormal and raises
/// FE_UNDERFLOW, which ActsExamples' FPE monitor reports as an error at a
/// per-event threshold of 0 -- so a fit that is numerically fine aborts the
/// whole job. Same constant and same reasoning as
/// Acts::detail::kLogMinNormalDouble in Chi2SurvivalFunction.hpp; duplicated
/// rather than included because this file is Core/src and that header is a
/// TrackFitting detail.
constexpr double kLogMinNormalDouble = -708.396418532264;

/// exp(x), or exactly 0 where exp would return a subnormal.
///
/// In the log-sum-exp below every argument is `something - maxLog`, i.e. <= 0,
/// so the largest term is exp(0) = 1 and this only ever truncates terms more
/// than ~708 log-units below it.
///
/// The cut has to be at the NORMAL/subnormal boundary, not at the point where
/// exp reaches zero: FE_UNDERFLOW is raised for a subnormal result too, so
/// cutting at log(smallest subnormal) would preserve the values and keep the
/// FPE. That means this is not bit-identical -- for x in [-745, -708.4] exp
/// returns ~1e-308 and this returns 0. Nothing downstream can see it: such a
/// term added to a denominator whose largest entry is 1 is a no-op in double
/// precision (1 + 1e-308 == 1), and as a weight, 1e-308 against 0 is
/// indistinguishable in a quantity bounded by 1.
///
/// Reached routinely rather than exceptionally: a wrong drift-circle mirror at
/// the cold end of the ladder has chi2/2T past 708 as soon as the fit has
/// converged onto the correct side, which is why 100 of 100 events tripped the
/// FPE monitor once the schedule was allowed to run past pass 6.
double expOrZero(double x) {
  return x < kLogMinNormalDouble ? 0. : std::exp(x);
}

}  // namespace

std::vector<double> Acts::AnnealingUtility::getWeights(
    State& state, const std::vector<double>& allChi2,
    const std::vector<double>* logNorms) const {
  const unsigned int idx = state.currentTemperatureIndex;
  const double currentInvTemp = 1. / (2. * m_cfg.setOfTemperatures[idx]);
  const std::size_t n = allChi2.size();

  std::vector<double> weights(n, 0.);
  if (n == 0) {
    return weights;
  }

  if (logNorms == nullptr) {
    // Norm-free form, bit-for-bit what the per-candidate getWeight above
    // produces -- kept so that callers which do not supply normalisations
    // (the vertexing, and any DAF run with the term switched off) are
    // completely unaffected by this overload existing.
    double denom = m_cfg.cutOffPerCandidate
                       ? static_cast<double>(n) * m_gaussCutTempVec[idx]
                       : m_gaussCutTempVec[idx];
    for (double val : allChi2) {
      denom += computeAnnealingWeight(val, currentInvTemp);
    }
    for (std::size_t j = 0; j < n; ++j) {
      weights[j] = computeAnnealingWeight(allChi2[j], currentInvTemp) / denom;
    }
    return weights;
  }

  assert(logNorms->size() == n &&
         "AnnealingUtility::getWeights: one log-normalisation per chi2");

  // GenFit's form. Everything is carried as a LOG term and exponentiated only
  // after the largest is factored out: norm = 1/sqrt((2pi)^d det S) underflows
  // to zero for a sub-micron 1D covariance and exp(-chi2/2T) overflows at the
  // cold end of the schedule, and the ratio is perfectly well defined in both
  // cases even though the individual terms are not.
  std::vector<double> logPhi(n);
  std::vector<double> logCut(n);
  double maxLog = -std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < n; ++j) {
    logPhi[j] = (*logNorms)[j] - allChi2[j] * currentInvTemp;
    logCut[j] = (*logNorms)[j] - m_cfg.cutOff * currentInvTemp;
    maxLog = std::max(maxLog, logPhi[j]);
    if (m_cfg.cutOffPerCandidate) {
      maxLog = std::max(maxLog, logCut[j]);
    }
  }
  if (!m_cfg.cutOffPerCandidate) {
    // One cut-off term for the surface. It still needs a normalisation to be
    // comparable with the candidate terms; the first candidate's is the only
    // defensible choice when the group does not share one, and this branch is
    // not GenFit's convention anyway -- cutOffPerCandidate is.
    maxLog = std::max(maxLog, logCut[0]);
  }
  if (!std::isfinite(maxLog)) {
    return weights;
  }

  double denom = 0.;
  for (std::size_t j = 0; j < n; ++j) {
    denom += expOrZero(logPhi[j] - maxLog);
  }
  if (m_cfg.cutOffPerCandidate) {
    for (std::size_t j = 0; j < n; ++j) {
      denom += expOrZero(logCut[j] - maxLog);
    }
  } else {
    denom += expOrZero(logCut[0] - maxLog);
  }
  if (!(denom > 0.) || !std::isfinite(denom)) {
    return weights;
  }
  // Normalise in LOG space rather than dividing. exp(x)/denom can be
  // subnormal even when exp(x) is not -- x just above the cut gives ~1e-308,
  // and denom is >= 1 because the largest term is exp(0) -- and a subnormal
  // raises FE_UNDERFLOW exactly like the exp did. Folding the division into
  // the exponent leaves one guarded exp, so every weight is either 0 or a
  // normal double. That also settles the two places downstream that consume
  // these weights (the |w - w_prev| in updateCandidateWeights, and w * chi2),
  // which underflowed on subnormal inputs of their own accord.
  const double logDenom = std::log(denom);
  for (std::size_t j = 0; j < n; ++j) {
    weights[j] = expOrZero(logPhi[j] - maxLog - logDenom);
  }
  return weights;
}

double Acts::AnnealingUtility::getWeight(State& state, double chi2) const {
  // Calculate 1/denominator in exp function
  const double currentInvTemp =
      1. / (2 * m_cfg.setOfTemperatures[state.currentTemperatureIndex]);

  return 1. /
         (1. + computeAnnealingWeight(m_cfg.cutOff - chi2, currentInvTemp));
}

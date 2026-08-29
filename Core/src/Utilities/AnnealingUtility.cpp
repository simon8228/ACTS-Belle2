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
    denom += std::exp(logPhi[j] - maxLog);
  }
  if (m_cfg.cutOffPerCandidate) {
    for (std::size_t j = 0; j < n; ++j) {
      denom += std::exp(logCut[j] - maxLog);
    }
  } else {
    denom += std::exp(logCut[0] - maxLog);
  }
  if (!(denom > 0.) || !std::isfinite(denom)) {
    return weights;
  }
  for (std::size_t j = 0; j < n; ++j) {
    weights[j] = std::exp(logPhi[j] - maxLog) / denom;
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

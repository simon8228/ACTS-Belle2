// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cmath>
#include <limits>

namespace Acts::detail {

/// log(DBL_MIN). Below this, std::exp underflows to a subnormal and raises
/// FE_UNDERFLOW -- which ActsExamples' FPE monitor treats as an error even
/// though the result is a correct, genuinely-zero probability. A DAF track
/// whose chi2 is large for its ndf reaches this routinely, so the exponent is
/// tested before it is exponentiated rather than after.
constexpr double kLogMinNormalDouble = -708.396418532264;

/// Regularised upper incomplete gamma Q(a, x) = Gamma(a, x) / Gamma(a).
///
/// Series expansion below the crossover and a continued fraction above it,
/// the standard split -- each converges quickly on one side and slowly on the
/// other. Self-contained so that a Core header does not have to pull in
/// boost::math for one scalar.
///
/// @param a Shape parameter, > 0
/// @param x Point to evaluate at, >= 0
inline double regularisedGammaQ(double a, double x) {
  if (!(a > 0.) || x < 0.) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (x == 0.) {
    return 1.;
  }

  const double logGammaA = std::lgamma(a);

  if (x < a + 1.) {
    // Series for the LOWER incomplete gamma P(a, x), then Q = 1 - P.
    double term = 1. / a;
    double sum = term;
    for (int n = 1; n < 1000; ++n) {
      term *= x / (a + n);
      sum += term;
      if (std::abs(term) < std::abs(sum) * 1e-15) {
        break;
      }
    }
    const double logPrefactor = -x + a * std::log(x) - logGammaA;
    if (sum <= 0.) {
      return 1.;
    }
    // P(a, x) = sum * exp(logPrefactor); if that underflows, Q = 1 - 0 = 1.
    const double logP = logPrefactor + std::log(sum);
    if (logP < kLogMinNormalDouble) {
      return 1.;
    }
    if (logPrefactor < kLogMinNormalDouble) {
      return 1. - std::exp(logP);
    }
    return 1. - sum * std::exp(logPrefactor);
  }

  // Lentz's continued fraction for Q(a, x) directly.
  constexpr double kTiny = 1e-300;
  double b = x + 1. - a;
  double c = 1. / kTiny;
  double d = 1. / b;
  double h = d;
  for (int n = 1; n < 1000; ++n) {
    const double an = -n * (n - a);
    b += 2.;
    d = an * d + b;
    if (std::abs(d) < kTiny) {
      d = kTiny;
    }
    c = b + an / c;
    if (std::abs(c) < kTiny) {
      c = kTiny;
    }
    d = 1. / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.) < 1e-15) {
      break;
    }
  }
  const double logPrefactor = -x + a * std::log(x) - logGammaA;
  if (h <= 0.) {
    return 0.;
  }
  const double logQ = logPrefactor + std::log(h);
  if (logQ < kLogMinNormalDouble) {
    return 0.;
  }
  if (logPrefactor < kLogMinNormalDouble) {
    // h > 1 rescues the product; avoid evaluating the underflowing factor.
    return std::exp(logQ);
  }
  return std::exp(logPrefactor) * h;
}

/// Survival function of the chi2 distribution, P(chi2 > value | ndf).
///
/// This is the "fit p-value": the probability that a correct fit would produce
/// a chi2 at least this large. GenFit's DAF uses it as a secondary convergence
/// test (DAF.cc:153, `status->getBackwardPVal()`), and basf2's DAFParameters
/// gates it on a strictly positive value.
///
/// @param chi2 The chi2 value, >= 0
/// @param ndf Degrees of freedom, > 0
/// @return The p-value in [0, 1], or NaN if the arguments are not usable
inline double chi2SurvivalFunction(double chi2, double ndf) {
  if (!(ndf > 0.) || !(chi2 >= 0.) || !std::isfinite(chi2)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return regularisedGammaQ(0.5 * ndf, 0.5 * chi2);
}

}  // namespace Acts::detail

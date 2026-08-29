// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"

#include <cmath>
#include <vector>

namespace Acts {
/// @brief Implements a deterministic thermodynamic annealing scheme
/// Ref. (1): CERN-THESIS-2010-027
class AnnealingUtility {
 public:
  /// @brief The annealing state
  /// Resetting the state is done by just creating a new instance
  struct State {
    /// Index pointing to current temperature value in configuration array
    unsigned int currentTemperatureIndex{0};

    /// Flag indicating whether equilibrium has been reached in annealing
    bool equilibriumReached{false};
  };

  /// @brief The configuration struct
  struct Config {
    Config();

    /// Constructor with parameters
    /// @param cutOff_ Cut-off threshold value
    /// @param setOfTemperatures_ Vector of temperature values for annealing
    Config(double cutOff_, std::vector<double> setOfTemperatures_)
        : cutOff(cutOff_), setOfTemperatures(std::move(setOfTemperatures_)) {}

    /// Insensitivity threshold for calculated weight at cutoff
    double cutOff{9.};

    /// Temperature sequence for annealing process, starts at first value
    /// and progresses towards the last value
    std::vector<double> setOfTemperatures{64., 16., 4., 2., 1.5, 1.};

    /// Count the cut-off term once PER COMPETING CANDIDATE rather than once
    /// per call, i.e. denominator = N*exp(-cutOff/2T) + sum_k exp(-chi2_k/2T).
    ///
    /// The cut-off term is the "none of these hypotheses is right" option. ACTS
    /// has always admitted one of them per weight evaluation; GenFit's DAF
    /// admits one per candidate -- its phi_cut accumulates inside the loop over
    /// measurements (DAF::calcWeights). For an N-candidate surface GenFit's
    /// outlier option is therefore N times as attractive, and the surface's
    /// retained weight sum correspondingly lower.
    ///
    /// This only bites while the cut-off term is comparable to the signal, i.e.
    /// at the HOT end of the schedule: for a CDC mirror pair at cutOff 10.83 it
    /// is a 33% reduction in the wire's total weight at T=100 and nothing at
    /// all by T=0.56, where exp(-cutOff/2T) has underflowed. Those early passes
    /// are what decide the basin the fit settles into, since each pass re-seeds
    /// the next.
    ///
    /// Not expressible as a different cutOff: matching it would need
    /// cutOff' = cutOff - 2T*ln(N), which moves with temperature and goes
    /// negative at the hot end.
    ///
    /// Default false, which leaves every existing caller -- notably
    /// AdaptiveMultiVertexFitter -- bit-for-bit unchanged.
    bool cutOffPerCandidate{false};
  };

  /// Constructor
  /// @param cfg The annealing configuration parameters
  explicit AnnealingUtility(const Config& cfg = Config()) : m_cfg(cfg) {
    // Set Gaussian cut-off terms for each temperature
    for (double temp : cfg.setOfTemperatures) {
      m_gaussCutTempVec.push_back(std::exp(-cfg.cutOff / (2. * temp)));
    }
  }

  /// Does the actual annealing step
  /// @param state The state object to perform annealing on
  void anneal(State& state) const;

  /// @brief Weight access
  ///
  /// @param state The state object
  /// @param chi2 Chi^2 for e.g. current track, i.e. compatibility
  /// of track to current vertex candidate
  /// @param allChi2 Vector of all chi^2 values, i.e. e.g. compatibilities
  /// of current track to all vertices it is currently attached to
  ///
  /// @return Calculated weight according to Eq.(5.46) in Ref.(1)
  double getWeight(State& state, double chi2,
                   const std::vector<double>& allChi2) const;

  /// @brief Weight access
  ///
  /// @param state The state object
  /// @param chi2 Chi^2
  ///
  /// @return Calculated weight
  double getWeight(State& state, double chi2) const;

  /// @brief Weights for a whole group of competing candidates, in one call.
  ///
  /// Equivalent to calling getWeight(state, allChi2[j], allChi2) for every j
  /// when @p logNorms is null -- same denominator, computed once instead of N
  /// times -- and the ONLY form that can reproduce GenFit's weights when it is
  /// not.
  ///
  /// GenFit multiplies every term, candidates and cut-off alike, by that
  /// candidate's own Gaussian prefactor norm_k = 1/sqrt((2*pi)^d * det S_k)
  /// (DAF.cc:313-321):
  ///
  ///   w_j = norm_j*exp(-chi2_j/2T)
  ///         / ( sum_k norm_k*exp(-chi2_k/2T) + sum_k norm_k*exp(-cutOff/2T) )
  ///
  /// Those prefactors cancel only when every candidate shares one S. GenFit's
  /// own comment flags that they need not ("the following assumes that in the
  /// competing hits could have different V"), and for CDC mirror pairs they do
  /// not: left and right get separately evaluated drift resolutions.
  ///
  /// @param state The annealing state, carrying the current temperature
  /// @param allChi2 chi2 of every candidate on this surface
  /// @param logNorms log(norm_k) per candidate, in the same order, or nullptr
  /// for the norm-free form. Must be the same length as @p allChi2.
  ///
  /// @return One weight per candidate, in the order of @p allChi2
  std::vector<double> getWeights(
      State& state, const std::vector<double>& allChi2,
      const std::vector<double>* logNorms = nullptr) const;

 private:
  /// Configuration object
  Config m_cfg;

  // For each temperature, a Gaussian term with the chi2 cut-off value
  // is calculated and stored here
  std::vector<double> m_gaussCutTempVec;
};
}  // namespace Acts

// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/SourceLink.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/PropagatorOptions.hpp"
#include "Acts/TrackFinding/CombinatorialKalmanFilterExtensions.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/detail/VoidFitterComponents.hpp"
#include "Acts/Utilities/AnnealingUtility.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/Delegate.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace Acts {

/// @addtogroup track_fitting
/// @{

/// The set of source links competing as hit hypotheses for a single surface.
/// Reuses CKF's small-vector cap since both CDC wire ambiguity groups and
/// silicon overlap groups are expected to be small.
using DafMeasurementGroup = CkfTypes::BranchVector<SourceLink>;

/// Extension struct which holds delegates to customise the DAF behavior.
/// @tparam traj_t The trajectory type
template <typename traj_t>
struct DafFitterExtensions {
  /// Type alias for track state proxy from trajectory
  using TrackStateProxy = typename traj_t::TrackStateProxy;
  /// Type alias for const track state proxy from trajectory
  using ConstTrackStateProxy = typename traj_t::ConstTrackStateProxy;

  /// Reused unmodified from the Kalman fitter: calibrates a single source
  /// link into a track state.
  using Calibrator = typename KalmanFitterExtensions<traj_t>::Calibrator;
  /// Reused unmodified from the Kalman fitter: incorporates the effective
  /// measurement into the track parameters.
  using Updater = typename KalmanFitterExtensions<traj_t>::Updater;
  /// Reused unmodified from the Kalman fitter: back-propagates measurement
  /// information along the track.
  using Smoother = typename KalmanFitterExtensions<traj_t>::Smoother;

  /// Computes the chi2 of one already-calibrated candidate against the
  /// (shared) predicted state, before the candidates are combined.
  using ChiSquareCalculator = Delegate<double(ConstTrackStateProxy)>;

  /// Combines N calibrated candidates and their annealed weights into a
  /// single effective measurement written into the real track state.
  using WeightedCombiner =
      Delegate<void(const std::vector<ConstTrackStateProxy>&,
                    const std::vector<double>&, TrackStateProxy)>;

  /// Retrieves the associated surface from a source link
  SourceLinkSurfaceAccessor surfaceAccessor;

  /// Calibrates each candidate source link, called once per candidate
  Calibrator calibrator;

  /// Computes the pre-update chi2 of a calibrated candidate
  ChiSquareCalculator chiSquareCalculator;

  /// Combines calibrated, weighted candidates into the effective measurement
  WeightedCombiner weightedCombiner;

  /// Incorporates the effective measurement into the track parameters
  Updater updater;

  /// Back-propagates measurement information along the track
  Smoother smoother;

  /// Default constructor which connects the default void components
  DafFitterExtensions() {
    surfaceAccessor.connect<&detail::voidSurfaceAccessor>();
    calibrator.template connect<&detail::voidFitterCalibrator<traj_t>>();
    chiSquareCalculator
        .template connect<&detail::voidDafChiSquareCalculator<traj_t>>();
    weightedCombiner
        .template connect<&detail::voidDafWeightedCombiner<traj_t>>();
    updater.template connect<&detail::voidFitterUpdater<traj_t>>();
    smoother.template connect<&detail::voidFitterSmoother<traj_t>>();
  }
};

/// Combined options for the Deterministic Annealing Filter.
/// @tparam traj_t The trajectory type
template <typename traj_t>
struct DafFitterOptions {
  /// @param gctx The geometry context for this fit
  /// @param mctx The magnetic context for this fit
  /// @param cctx The calibration context for this fit
  /// @param extensions_ The DAF extensions
  /// @param pOptions The plain propagator options
  /// @param tSurface The target surface for the fit
  /// @param mScattering Whether to include multiple scattering
  /// @param eLoss Whether to include energy loss
  /// @param freeToBoundCorrection_ Correction for non-linearity effect during transform from free to bound
  /// @param annealingConfig_ The annealing temperature schedule and cutoff
  /// @param iterationsPerTemperature_ Number of forward+smooth passes run at each temperature before annealing
  /// @param maxIterations_ Hard cap on the total number of forward+smooth passes across the whole schedule
  DafFitterOptions(const GeometryContext& gctx,
                   const MagneticFieldContext& mctx,
                   std::reference_wrapper<const CalibrationContext> cctx,
                   DafFitterExtensions<traj_t> extensions_,
                   const PropagatorPlainOptions& pOptions,
                   const Surface* tSurface = nullptr, bool mScattering = true,
                   bool eLoss = true,
                   const FreeToBoundCorrection& freeToBoundCorrection_ =
                       FreeToBoundCorrection(false),
                   AnnealingUtility::Config annealingConfig_ =
                       AnnealingUtility::Config(),
                   std::size_t iterationsPerTemperature_ = 1,
                   std::size_t maxIterations_ = 100)
      : geoContext(gctx),
        magFieldContext(mctx),
        calibrationContext(cctx),
        extensions(extensions_),
        propagatorPlainOptions(pOptions),
        referenceSurface(tSurface),
        multipleScattering(mScattering),
        energyLoss(eLoss),
        freeToBoundCorrection(freeToBoundCorrection_),
        annealingConfig(std::move(annealingConfig_)),
        iterationsPerTemperature(iterationsPerTemperature_),
        maxIterations(maxIterations_) {}

  /// Context object for the geometry
  std::reference_wrapper<const GeometryContext> geoContext;
  /// Context object for the magnetic field
  std::reference_wrapper<const MagneticFieldContext> magFieldContext;
  /// Context object for the calibration
  std::reference_wrapper<const CalibrationContext> calibrationContext;

  /// Extensions for calibration, weighting and combination
  DafFitterExtensions<traj_t> extensions;

  /// The trivial propagator options
  PropagatorPlainOptions propagatorPlainOptions;

  /// The reference surface
  const Surface* referenceSurface = nullptr;

  /// Strategy to propagate to reference surface
  TrackExtrapolationStrategy referenceSurfaceStrategy =
      TrackExtrapolationStrategy::firstOrLast;

  /// Whether to consider multiple scattering
  bool multipleScattering = true;

  /// Whether to consider energy loss
  bool energyLoss = true;

  /// Whether to include non-linear correction during global to local
  /// transformation
  FreeToBoundCorrection freeToBoundCorrection;

  /// The annealing temperature schedule and cutoff
  AnnealingUtility::Config annealingConfig;

  /// Number of forward+smooth passes run at each temperature before
  /// annealing to the next one. Fixed count for M1; no weight-delta-based
  /// early stopping yet.
  std::size_t iterationsPerTemperature = 1;

  /// Hard cap on the total number of forward+smooth passes across the whole
  /// schedule, as a safety net independent of the temperature schedule
  /// length.
  std::size_t maxIterations = 100;
};

/// Per-surface persistent annealing state: the fixed candidate source links,
/// their current annealed weight, and their most recent chi2 -- carried by
/// the outer driver across forward+smooth passes since it cannot live in the
/// trajectory (only the effective combined measurement is written there).
struct DafCandidateState {
  /// The candidate source links, fixed for the whole fit
  DafMeasurementGroup candidates;
  /// Current annealed weight of each candidate, same order as candidates
  std::vector<double> weights;
  /// Most recent chi2 of each candidate, same order as candidates
  std::vector<double> chi2;
};

/// Per-surface map of persistent annealing state, keyed the same way as the
/// input measurement groups.
using DafCandidateMap = std::unordered_map<const Surface*, DafCandidateState>;

/// @}

}  // namespace Acts

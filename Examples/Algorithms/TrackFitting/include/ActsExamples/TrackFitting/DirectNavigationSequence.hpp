// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryContext.hpp"

#include <vector>

namespace Acts {
class Surface;
}

namespace ActsExamples {

/// Prepare a DirectNavigator surface sequence for a fit: inject the material
/// surfaces reachable within the hit span, place them among the hits, and drop
/// duplicates. Only material from the first hit onwards is injected: the
/// stretch between the seed and the first hit is not assumed to be flown (see
/// the implementation for why this matters for displaced tracks). Shared by
/// the KF and (in KF-equivalence mode) the DAF fitter functions so both fits
/// traverse the identical sequence.
///
/// Everything is ordered by path from the seed. The DirectNavigator walks the
/// sequence blindly, so the order must be monotonic along the trajectory --
/// otherwise it backtracks and exhausts the step budget (PropagatorError:2).
///
/// Paths are measured along the helix when @p signedHelixRadius is given.
/// Ordering by the straight-line ray instead -- as this function used to do --
/// is only valid while the trajectory is nearly straight: for a low-pT curler
/// the seed tangent bears no relation to the real path, which scrambles the
/// sequence and collapses fit efficiency in the lowest pT bins.
///
/// @param geoContext The geometry context
/// @param seedPosition Seed position the paths are measured from
/// @param seedDirection Seed direction at @p seedPosition
/// @param surfaceSequence In: the hit surfaces in traversal order; out: the
///        same order with material surfaces placed among them, deduplicated
/// @param materialSurfaces All surfaces of the geometry carrying material,
///        candidates for injection
/// @param signedHelixRadius Signed transverse radius of the seed helix, sign
///        giving the turning sense. Pass 0 (the default) for no curvature
///        information, which falls back to straight-line paths.
/// @param droppedUnreachable When non-null and the reachability cut is enabled
///        (ACTS_DNS_MAXREACH), receives the surfaces removed from the sequence
///        because the seed helix cannot reach their radius. The caller MUST
///        erase these from its measurement groups as well: DafActor finishes a
///        pass on measurementStates == inputMeasurementGroups.size(), so a
///        group left in place but absent from the sequence is never satisfied
///        and the propagation runs on to the path limit.
/// @return The path length spanned by the prepared sequence, for the caller to
///         size the propagator's path limit with. A fixed limit truncates
///         curlers, whose arc length grows as the momentum falls.
double prepareDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const Acts::Vector3& seedPosition, const Acts::Vector3& seedDirection,
    std::vector<const Acts::Surface*>& surfaceSequence,
    const std::vector<const Acts::Surface*>& materialSurfaces,
    double signedHelixRadius = 0.,
    std::vector<const Acts::Surface*>* droppedUnreachable = nullptr);

/// One surface a fit pass actually reached, for resortDirectNavigationSequence.
struct ReachedTrajectoryPoint {
  const Acts::Surface* surface = nullptr;
  /// Path length at which the pass reached it, from where the pass started
  double pathLength = 0.;
  /// Fitted (smoothed) position and direction there
  Acts::Vector3 position = Acts::Vector3::Zero();
  Acts::Vector3 direction = Acts::Vector3::UnitZ();
  /// Signed transverse helix radius of the fitted track there (see
  /// prepareDirectNavigationSequence); 0 for no curvature information
  double signedHelixRadius = 0.;
};

/// Re-order a DirectNavigator sequence from the trajectory a fit pass
/// actually propagated, GenFit-style, instead of from a seed helix.
///
/// A surface the pass reached goes where it was reached (its path length). A
/// surface it did not reach -- the propagation ended first, or the navigator
/// stepped past it -- goes where the FITTED track puts it: the helix of the
/// fitted state on the last reached surface before it in @p previousSequence
/// (its anchor), with that state's own theta, z and curvature. Consecutive
/// unreached surfaces behind one anchor are turn-unwrapped the same way as in
/// prepareDirectNavigationSequence. Nothing is dropped: a skipped surface is
/// back in the queue at its estimated place.
///
/// @param geoContext The geometry context
/// @param previousSequence The sequence the pass navigated
/// @param reached The surfaces it reached, in trajectory order
/// @return The re-ordered sequence, same surfaces (same multiplicity)
std::vector<const Acts::Surface*> resortDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const std::vector<const Acts::Surface*>& previousSequence,
    const std::vector<ReachedTrajectoryPoint>& reached);

}  // namespace ActsExamples

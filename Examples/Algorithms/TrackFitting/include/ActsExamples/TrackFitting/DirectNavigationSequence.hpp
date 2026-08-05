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
/// duplicates. Shared by the KF and (in KF-equivalence mode) the DAF fitter
/// functions so both fits traverse the identical sequence.
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
/// @return The path length spanned by the prepared sequence, for the caller to
///         size the propagator's path limit with. A fixed limit truncates
///         curlers, whose arc length grows as the momentum falls.
double prepareDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const Acts::Vector3& seedPosition, const Acts::Vector3& seedDirection,
    std::vector<const Acts::Surface*>& surfaceSequence,
    const std::vector<const Acts::Surface*>& materialSurfaces,
    double signedHelixRadius = 0.);

}  // namespace ActsExamples

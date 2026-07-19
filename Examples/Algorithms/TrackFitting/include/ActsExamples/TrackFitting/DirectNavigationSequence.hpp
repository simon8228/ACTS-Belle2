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
/// surfaces reachable within the hit span (straight-line path below the
/// farthest hit surface + 1 mm), sort everything by straight-line path from
/// the seed, and drop duplicates. Shared by the KF and (in KF-equivalence
/// mode) the DAF fitter functions so both fits traverse the identical
/// sequence.
///
/// @param geoContext The geometry context
/// @param seedPosition Seed position the straight-line paths are measured from
/// @param seedDirection Seed direction defining the straight line
/// @param surfaceSequence In: the hit surfaces; out: sorted, deduplicated,
///        with material surfaces injected
/// @param materialSurfaces All surfaces of the geometry carrying material,
///        candidates for injection
void prepareDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const Acts::Vector3& seedPosition, const Acts::Vector3& seedDirection,
    std::vector<const Acts::Surface*>& surfaceSequence,
    const std::vector<const Acts::Surface*>& materialSurfaces);

}  // namespace ActsExamples

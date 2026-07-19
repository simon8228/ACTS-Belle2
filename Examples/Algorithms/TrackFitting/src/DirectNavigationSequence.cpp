// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFitting/DirectNavigationSequence.hpp"

#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Intersection.hpp"

#include <algorithm>

namespace ActsExamples {

void prepareDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const Acts::Vector3& seedPosition, const Acts::Vector3& seedDirection,
    std::vector<const Acts::Surface*>& surfaceSequence,
    const std::vector<const Acts::Surface*>& materialSurfaces) {
  auto pathTo = [&](const Acts::Surface* srf) -> double {
    auto multiIntersection =
        srf->intersect(geoContext, seedPosition, seedDirection,
                       Acts::BoundaryTolerance::Infinite());
    const auto& intersection = multiIntersection.closestForward();
    if (intersection.status() >= Acts::IntersectionStatus::reachable &&
        intersection.pathLength() > 0) {
      return intersection.pathLength();
    }
    // No forward intersection of the ray (e.g. nearly parallel
    // surface): fall back to the projection of the surface center.
    return (srf->center(geoContext) - seedPosition).dot(seedDirection);
  };
  double maxHitPath = 0.;
  for (const Acts::Surface* srf : surfaceSequence) {
    auto hitIntersection =
        srf->intersect(geoContext, seedPosition, seedDirection,
                       Acts::BoundaryTolerance::None());
    const auto& hitIx = hitIntersection.closestForward();
    if (hitIx.status() >= Acts::IntersectionStatus::reachable &&
        hitIx.pathLength() > 0) {
      maxHitPath = std::max(maxHitPath, hitIx.pathLength());
    }
  }
  for (const Acts::Surface* srf : materialSurfaces) {
    auto multiIntersection =
        srf->intersect(geoContext, seedPosition, seedDirection,
                       Acts::BoundaryTolerance::None());
    const auto& intersection = multiIntersection.closestForward();
    if (intersection.status() >= Acts::IntersectionStatus::reachable &&
        intersection.pathLength() > 0 &&
        intersection.pathLength() < maxHitPath + 1.0) {
      surfaceSequence.push_back(srf);
    }
  }

  std::stable_sort(surfaceSequence.begin(), surfaceSequence.end(),
                   [&](const Acts::Surface* a, const Acts::Surface* b) {
                     return pathTo(a) < pathTo(b);
                   });
  surfaceSequence.erase(
      std::unique(surfaceSequence.begin(), surfaceSequence.end()),
      surfaceSequence.end());
}

}  // namespace ActsExamples

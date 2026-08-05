// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFitting/DirectNavigationSequence.hpp"

#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Intersection.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace ActsExamples {

namespace {
/// Below this transverse radius a surface centre counts as sitting on the beam
/// axis, i.e. the surface is a concentric cylinder and has to be ordered by the
/// radius at which the trajectory crosses it rather than by its centre.
constexpr double kOnAxisTolerance = 1e-3;

/// Diagnostics, all opt-in via the environment so the default build behaves
/// exactly as configured in code:
///   ACTS_DNS_HELIX=0  -- ignore the seed curvature, order by straight-line
///                        ray (the historical behaviour), for A/B testing
///   ACTS_DNS_DEBUG=N  -- dump the prepared sequence for the first N calls,
///                        so its monotonicity can be inspected directly
bool dnsUseHelix() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DNS_HELIX");
    return env == nullptr || std::atoi(env) != 0;
  }();
  return value;
}

int dnsDebugBudget() {
  static const int value = [] {
    const char* env = std::getenv("ACTS_DNS_DEBUG");
    return env == nullptr ? 0 : std::atoi(env);
  }();
  return value;
}

std::atomic<int> dnsDebugDone{0};
}  // namespace

double prepareDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const Acts::Vector3& seedPosition, const Acts::Vector3& seedDirection,
    std::vector<const Acts::Surface*>& surfaceSequence,
    const std::vector<const Acts::Surface*>& materialSurfaces,
    double signedHelixRadius) {
  const double sinTheta = seedDirection.head<2>().norm();
  // A vanishing/infinite radius means "no curvature information": fall back to
  // the straight-line ray, which is what this function used to do always.
  const bool curved = dnsUseHelix() && std::isfinite(signedHelixRadius) &&
                      std::abs(signedHelixRadius) > 0. && sinTheta > 1e-6;

  // ACTS' signed radius rho = sin(theta)/((q/p)*Bz) places the helix centre at
  // pos + rho * (sin(phi), -cos(phi)), i.e. the direction rotated by MINUS 90
  // deg (see Acts::ImpactPointEstimator). The normal below is the PLUS 90 deg
  // rotation, so flipping the sign once here makes both the centre and the
  // turning sense come out right; getting this backwards mirrors every arc
  // angle to 2*pi - phi and reverses the ordering.
  const double radius = -signedHelixRadius;

  // Transverse circle of the helix.
  const double rho = std::abs(radius);
  const double sense = (radius >= 0.) ? 1. : -1.;
  Acts::Vector2 centre = Acts::Vector2::Zero();
  double phiStart = 0.;
  double centreDist = 0.;
  if (curved) {
    const Acts::Vector2 dirT = seedDirection.head<2>() / sinTheta;
    const Acts::Vector2 normal(-dirT.y(), dirT.x());
    centre = seedPosition.head<2>() + radius * normal;
    const Acts::Vector2 fromCentre = seedPosition.head<2>() - centre;
    phiStart = std::atan2(fromCentre.y(), fromCentre.x());
    centreDist = centre.norm();
  }

  // Arc length from the seed to the helix point whose transverse projection
  // lines up with `point`. Wrapped into one turn, so for a curler that crosses
  // a surface twice this is the first crossing.
  auto helixPathToPoint = [&](const Acts::Vector2& point) {
    const Acts::Vector2 u0 = seedPosition.head<2>() - centre;
    const Acts::Vector2 up = point - centre;
    double dphi = sense * std::atan2(u0.x() * up.y() - u0.y() * up.x(),
                                     u0.dot(up));
    if (dphi < 0.) {
      dphi += 2. * std::numbers::pi;
    }
    return rho * dphi / sinTheta;
  };

  // Arc length to the first crossing of the transverse radius `rt`. Solves
  // |centre + rho*u(phi)|^2 = rt^2, which has no solution when the helix never
  // reaches that radius -- an honest "unreachable" rather than a bogus order.
  auto helixPathToRadius = [&](double rt) {
    if (centreDist < kOnAxisTolerance) {
      return std::numeric_limits<double>::infinity();
    }
    const double cosArg =
        (rt * rt - centreDist * centreDist - rho * rho) / (2. * centreDist * rho);
    if (std::abs(cosArg) > 1.) {
      return std::numeric_limits<double>::infinity();
    }
    const double gamma = std::atan2(centre.y(), centre.x());
    const double offset = std::acos(cosArg);
    double best = std::numeric_limits<double>::infinity();
    for (const double sign : {1., -1.}) {
      double dphi = sense * (gamma + sign * offset - phiStart);
      dphi = std::fmod(dphi, 2. * std::numbers::pi);
      if (dphi < 0.) {
        dphi += 2. * std::numbers::pi;
      }
      best = std::min(best, rho * dphi / sinTheta);
    }
    return best;
  };

  auto pathTo = [&](const Acts::Surface* srf) -> double {
    if (curved) {
      const Acts::Vector3 srfCentre = srf->center(geoContext);
      if (srfCentre.head<2>().norm() > kOnAxisTolerance) {
        // Off-axis: straws, planes, discs -- the centre is a real transverse
        // point, so the arc length to it is unambiguous.
        return helixPathToPoint(srfCentre.head<2>());
      }
      if (const auto* cyl = dynamic_cast<const Acts::CylinderSurface*>(srf)) {
        // Concentric cylinder: its centre carries no phi information, so order
        // it by the radius at which the helix crosses it.
        return helixPathToRadius(cyl->bounds().get(Acts::CylinderBounds::eR));
      }
    }
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

  // Everything is ordered by path from the seed, exactly as before -- the
  // DirectNavigator walks the sequence blindly, so an order that is not
  // monotonic along the trajectory makes it backtrack and burn the whole step
  // budget (PropagatorError:2). The only change is the metric: along the helix
  // when the curvature is known, because for a low-pT curler the straight-line
  // ray bears no relation to the real trajectory.
  std::vector<std::pair<double, const Acts::Surface*>> ordered;
  ordered.reserve(surfaceSequence.size() + materialSurfaces.size());

  const std::size_t hitCount = surfaceSequence.size();
  double maxHitPath = 0.;
  for (const Acts::Surface* srf : surfaceSequence) {
    const double path = pathTo(srf);
    if (std::isfinite(path) && path > 0.) {
      maxHitPath = std::max(maxHitPath, path);
    }
    ordered.emplace_back(path, srf);
  }
  // Selection and ordering are two different questions and must not share a
  // test. `pathTo` deliberately ignores bounds so it can return a path for
  // every surface, which is what ordering needs; deciding *whether* a material
  // surface belongs in the sequence needs the opposite -- a bounds-constrained
  // intersection, so only surfaces the trajectory actually crosses are taken.
  // Without this the CDC-only fit picked up every SVD module and PXD cylinder
  // in the geometry (47 hits -> ~200 surfaces), and the DirectNavigator burned
  // its whole step budget walking surfaces that are not on the trajectory.
  auto crossesWithinBounds = [&](const Acts::Surface* srf) {
    auto multiIntersection =
        srf->intersect(geoContext, seedPosition, seedDirection,
                       Acts::BoundaryTolerance::None());
    const auto& intersection = multiIntersection.closestForward();
    return intersection.status() >= Acts::IntersectionStatus::reachable &&
           intersection.pathLength() > 0;
  };

  for (const Acts::Surface* srf : materialSurfaces) {
    if (!crossesWithinBounds(srf)) {
      continue;
    }
    const double path = pathTo(srf);
    if (std::isfinite(path) && path > 0. && path < maxHitPath + 1.0) {
      ordered.emplace_back(path, srf);
    }
  }

  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });

  surfaceSequence.clear();
  std::vector<double> keptPaths;
  double maxPath = 0.;
  for (const auto& [path, srf] : ordered) {
    if (surfaceSequence.empty() || surfaceSequence.back() != srf) {
      surfaceSequence.push_back(srf);
      keptPaths.push_back(path);
      if (std::isfinite(path)) {
        maxPath = std::max(maxPath, path);
      }
    }
  }

  if (dnsDebugBudget() > 0 && dnsDebugDone.fetch_add(1) < dnsDebugBudget()) {
    std::fprintf(stderr,
                 "[DNS] curved=%d rho=%.1f sinTheta=%.4f seed=(%.1f,%.1f,%.1f) "
                 "nHit=%zu nSeq=%zu span=%.1f\n",
                 curved ? 1 : 0, rho, sinTheta, seedPosition.x(),
                 seedPosition.y(), seedPosition.z(), hitCount,
                 surfaceSequence.size(), maxPath);
    for (std::size_t i = 0; i < surfaceSequence.size(); ++i) {
      const Acts::Vector3 c = surfaceSequence[i]->center(geoContext);
      std::fprintf(stderr, "[DNS]   %3zu path=%10.1f r=%8.1f z=%9.1f vol=%u lay=%u sens=%u\n",
                   i, keptPaths[i], c.head<2>().norm(), c.z(),
                   static_cast<unsigned>(surfaceSequence[i]->geometryId().volume()),
                   static_cast<unsigned>(surfaceSequence[i]->geometryId().layer()),
                   static_cast<unsigned>(surfaceSequence[i]->geometryId().sensitive()));
    }
  }
  return maxPath;
}

}  // namespace ActsExamples

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
#include "Acts/Surfaces/LineBounds.hpp"
#include "Acts/Surfaces/LineSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Intersection.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <limits>
#include <unordered_map>
#include <numbers>
#include <utility>
#include <vector>

namespace ActsExamples {

namespace {
/// Below this transverse radius a surface centre counts as sitting on the beam
/// axis, i.e. the surface is a concentric cylinder and has to be ordered by the
/// radius at which the trajectory crosses it rather than by its centre.
constexpr double kOnAxisTolerance = 1e-3;

/// Slack on the "material only past the first hit" cut, so a material surface
/// that sits at the same path as the first measured surface (e.g. a layer
/// cylinder through the innermost wires) is not lost to rounding.
constexpr double kPreHitMaterialTolerance = 1.;

/// Diagnostics and A/B switches, all opt-in via the environment so the default
/// build behaves exactly as configured in code:
///   ACTS_DNS_HELIX=0    -- ignore the seed curvature, order by straight-line
///                          ray (the historical behaviour), for A/B testing
///   ACTS_DNS_UNWRAP=0   -- disable turn unwrapping, i.e. keep every path
///                          folded into the first turn (the behaviour before
///                          that fix), for A/B testing
///   ACTS_DNS_HELIXSEL=0 -- select material surfaces with the straight-line
///                          ray from the seed instead of the trajectory (the
///                          behaviour before that fix), for A/B testing
///   ACTS_DNS_DEBUG=N    -- dump the prepared sequence for the first N calls,
///                          so its monotonicity can be inspected directly
bool dnsUseHelix() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DNS_HELIX");
    return env == nullptr || std::atoi(env) != 0;
  }();
  return value;
}

bool dnsUnwrapTurns() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DNS_UNWRAP");
    return env == nullptr || std::atoi(env) != 0;
  }();
  return value;
}

/// Reachability cut: drop sequence surfaces whose radius exceeds this multiple
/// of the largest radius the seed helix can reach. 0 (the default) disables it.
///
/// A track from the perigee cannot exceed r = |centre| + rho. Measured on a
/// 10k-event CDC fit, 25.7% of sub-200 MeV sequences contained surfaces beyond
/// that, and the DirectNavigator -- which walks the sequence blindly -- burns
/// its whole step budget propagating toward one it can never intersect
/// (PropagatorError:2, 84 of 127 failures, immune to maxSteps=15000). The
/// distribution of rmax/(2 rho) separates cleanly: p75 = 1.04, then p90 = 1.76,
/// p99 = 3.04. A factor of ~1.5 therefore leaves seed-consistent tracks alone
/// and cuts only the pathological tail (14.3% of the low-pT class). The seed's
/// own rho is the underlying error -- the hits out there are real -- so the
/// factor must stay generous.
double dnsMaxReachFactor() {
  static const double value = [] {
    const char* env = std::getenv("ACTS_DNS_MAXREACH");
    return env == nullptr ? 0. : std::strtod(env, nullptr);
  }();
  return value;
}

bool dnsHelixSelection() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DNS_HELIXSEL");
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
    double signedHelixRadius,
    std::vector<const Acts::Surface*>* droppedUnreachable) {
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

  // Largest transverse radius the seed helix reaches, and the cut derived from
  // it. Only meaningful for a curved trajectory; a straight one is unbounded.
  const double reachFactor = dnsMaxReachFactor();
  const double maxReachR =
      (curved && reachFactor > 0.)
          ? reachFactor * (centreDist + rho)
          : std::numeric_limits<double>::infinity();

  // Transverse radius of a surface. A cylinder concentric with the beam has its
  // centre on the axis, so its radius is the bounds value, not the centre.
  auto surfaceRadius = [&](const Acts::Surface* srf) {
    const double rc = srf->center(geoContext).head<2>().norm();
    if (rc > kOnAxisTolerance) {
      return rc;
    }
    if (const auto* cyl = dynamic_cast<const Acts::CylinderSurface*>(srf)) {
      return cyl->bounds().get(Acts::CylinderBounds::eR);
    }
    return rc;
  };

  auto unreachable = [&](const Acts::Surface* srf) {
    return std::isfinite(maxReachR) && surfaceRadius(srf) > maxReachR;
  };

  // Path length of one full turn. Everything `helixPathToPoint` returns is
  // folded into [0, turnLength): the unwrapping below adds back the turns.
  const double turnLength = curved ? 2. * std::numbers::pi * rho / sinTheta
                                   : std::numeric_limits<double>::infinity();

  // Position and direction on the seed helix at arc length `s`. Used to place
  // the trajectory where the bounds check needs it, so that deciding *whether*
  // a surface is crossed uses the same geometry as deciding *where* it sits.
  auto helixStateAt = [&](double s) {
    const double ang = phiStart + sense * s * sinTheta / rho;
    const Acts::Vector3 pos(centre.x() + rho * std::cos(ang),
                            centre.y() + rho * std::sin(ang),
                            seedPosition.z() + s * seedDirection.z());
    const Acts::Vector3 dir(-sense * std::sin(ang) * sinTheta,
                            sense * std::cos(ang) * sinTheta,
                            seedDirection.z());
    return std::make_pair(pos, dir);
  };

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

  // A straw's transverse position is constant along the wire only when the
  // wire runs parallel to z. A CDC stereo wire does not: its midpoint sits
  // tens of mm (transversely) from where the track actually crosses it, so
  // ordering such a surface by its centre places it at the wrong path and the
  // DirectNavigator -- which walks the sequence blindly -- can step straight
  // past it. Measured on a steep track: the midpoint is 23-81 mm from the
  // crossing, and 13% of stereo surfaces never got filtered as a result.
  //
  // Solve for the crossing by fixed-point iteration path -> z -> wire point
  // -> path. It converges in a couple of steps because the wire's transverse
  // drift over one iteration's change in z is small compared to the cell.
  auto strawCrossingXY = [&](const Acts::LineSurface* line) -> Acts::Vector2 {
    const Acts::Vector3 wireCentre = line->center(geoContext);
    const Acts::Vector3 wireDir = line->lineDirection(geoContext);
    Acts::Vector2 xy = wireCentre.head<2>();
    // An axial wire (or one perpendicular to z, which the CDC has none of)
    // has nothing to resolve: its transverse position is already the answer.
    if (std::abs(wireDir.z()) < 1e-6 ||
        wireDir.head<2>().norm() < kOnAxisTolerance) {
      return xy;
    }
    double halfLength = std::numeric_limits<double>::infinity();
    if (const auto* lb =
            dynamic_cast<const Acts::LineBounds*>(&line->bounds())) {
      halfLength = lb->get(Acts::LineBounds::eHalfLengthZ);
    }
    const double cosTheta = seedDirection.z();
    for (int i = 0; i < 3; ++i) {
      const double path = helixPathToPoint(xy);
      if (!std::isfinite(path)) {
        break;
      }
      const double z = seedPosition.z() + path * cosTheta;
      // Clamped to the physical wire: past the endplate the nearest point of
      // the wire is its end, which is the honest thing to order by.
      const double t = std::clamp((z - wireCentre.z()) / wireDir.z(),
                                  -halfLength, halfLength);
      xy = wireCentre.head<2>() + t * wireDir.head<2>();
    }
    return xy;
  };

  auto pathTo = [&](const Acts::Surface* srf) -> double {
    if (curved) {
      if (const auto* line = dynamic_cast<const Acts::LineSurface*>(srf)) {
        const Acts::Vector2 xy = strawCrossingXY(line);
        if (xy.norm() > kOnAxisTolerance) {
          return helixPathToPoint(xy);
        }
      }
      const Acts::Vector3 srfCentre = srf->center(geoContext);
      if (srfCentre.head<2>().norm() > kOnAxisTolerance) {
        // Off-axis: planes, discs -- the centre is a real transverse point,
        // so the arc length to it is unambiguous.
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

  // Turn unwrapping. `helixPathToPoint` folds every path into the first turn,
  // so a curler that makes more than one revolution has its later hits ordered
  // as though they came on turn one -- and, worse, the span returned below
  // (which sizes the propagator's path limit) never exceeds a single turn, so
  // the propagation is cut short and every surface past that point is dropped
  // without even being recorded as a hole.
  //
  // The incoming order is the proto track's, i.e. the pattern-recognised
  // traversal order, so the turn count can simply be accumulated along it: a
  // wrapped path that jumps backwards by more than half a turn is a new
  // revolution. Half a turn as the threshold is the standard phase-unwrap
  // rule and tolerates the small non-monotonicity of neighbouring hits.
  const bool unwrap = curved && dnsUnwrapTurns() && std::isfinite(turnLength);
  const std::size_t hitCount = surfaceSequence.size();
  double maxHitPath = 0.;
  double minHitPath = std::numeric_limits<double>::infinity();
  double turnOffset = 0.;
  double prevWrapped = 0.;
  bool havePrev = false;
  for (const Acts::Surface* srf : surfaceSequence) {
    if (unreachable(srf)) {
      // Reported back so the caller can drop the matching measurement group;
      // see the header. Losing the hit is the point: it converts a whole track
      // lost to a step-budget burn into a track fitted with fewer hits.
      if (droppedUnreachable != nullptr) {
        droppedUnreachable->push_back(srf);
      }
      continue;
    }
    double path = pathTo(srf);
    if (unwrap && std::isfinite(path)) {
      if (havePrev && path < prevWrapped - 0.5 * turnLength) {
        turnOffset += turnLength;
      }
      prevWrapped = path;
      havePrev = true;
      path += turnOffset;
    }
    if (std::isfinite(path) && path > 0.) {
      maxHitPath = std::max(maxHitPath, path);
      minHitPath = std::min(minHitPath, path);
    }
    ordered.emplace_back(path, srf);
  }
  // How many revolutions the hits span, so material surfaces can be repeated
  // on each of them below.
  const std::size_t nTurns =
      (unwrap && turnLength > 0.)
          ? static_cast<std::size_t>(std::floor(maxHitPath / turnLength))
          : 0;
  // Material is only placed from the first measured surface onwards. The seed
  // sits on the perigee at the origin, but what it describes is the track the
  // finder saw in the CDC: nothing says the particle ever flew the stretch
  // between the perigee and its first hit. For a primary that stretch is real,
  // but the seed momentum is measured from the CDC hits, i.e. already after
  // that material, so taking energy off it again on the way out biases the
  // prior. For a displaced track it is not real at all -- a particle born at
  // r = 400 mm was being scattered and slowed in the beam pipe, PXD, SVD and
  // CDC inner wall along a helix extension it never travelled, up to a metre
  // of arc for a soft track. On BBbar the ACTS-vs-GenFit efficiency gap grew
  // with the length of that extension, to 10-12 points beyond 300 mm. The
  // material inside the first hit is still accounted for on the extrapolation
  // to the perigee, which runs with a MaterialInteractor.
  //
  // With no hit path known (every hit unreachable or non-positive) nothing is
  // cut, as before.
  const double materialStartPath =
      std::isfinite(minHitPath) ? minHitPath - kPreHitMaterialTolerance : 0.;

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

  // ...but shooting that ray from the *seed* asks whether the seed tangent
  // crosses the surface, not whether the track does, and the two part company
  // fast: at 1.5 T a 1 GeV track is already 221 mm off its own tangent after a
  // metre of path, 620 mm at 300 MeV -- against an 18 mm CDC cell. So the ray
  // both admits material the track never reaches (which the DirectNavigator
  // then spends its step budget chasing, the same failure the bounds check
  // above was added to stop) and rejects material it does cross (whose
  // scattering and energy loss are then simply never applied).
  //
  // Test on the trajectory instead: step back a little from the crossing the
  // ordering computed, and intersect from there. Over that short distance the
  // helix and its local tangent agree, so this is the same intersection
  // machinery evaluated where the track actually is.
  constexpr double kLinearisationStep = 10.;
  auto crossesOnTrajectory = [&](const Acts::Surface* srf, double path) {
    if (!curved || !dnsHelixSelection() || !std::isfinite(path)) {
      return crossesWithinBounds(srf);
    }
    const auto [pos, dir] = helixStateAt(std::max(0., path - kLinearisationStep));
    auto multiIntersection =
        srf->intersect(geoContext, pos, dir, Acts::BoundaryTolerance::None());
    const auto& intersection = multiIntersection.closestForward();
    return intersection.status() >= Acts::IntersectionStatus::reachable &&
           intersection.pathLength() > 0;
  };

  for (const Acts::Surface* srf : materialSurfaces) {
    if (unreachable(srf)) {
      continue;   // material only; nothing to erase from the measurement groups
    }
    const double wrapped = pathTo(srf);
    if (!std::isfinite(wrapped) || wrapped <= 0.) {
      continue;
    }
    // A curler crosses the same material surface once per revolution, and each
    // crossing is a real interaction, so the surface is placed on every turn
    // the hits span. nTurns is 0 unless the track actually wraps, leaving the
    // single-turn case byte-for-byte as it was.
    for (std::size_t turn = 0; turn <= nTurns; ++turn) {
      // Not `wrapped + turn * turnLength`: turnLength is infinite in the
      // straight-line case, and 0 * infinity is NaN, which would silently
      // discard every material surface there.
      const double path =
          (turn == 0) ? wrapped
                      : wrapped + static_cast<double>(turn) * turnLength;
      if (!(path > 0.) || path < materialStartPath ||
          path >= maxHitPath + 1.0) {
        continue;
      }
      if (!crossesOnTrajectory(srf, path)) {
        continue;
      }
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

std::vector<const Acts::Surface*> resortDirectNavigationSequence(
    const Acts::GeometryContext& geoContext,
    const std::vector<const Acts::Surface*>& previousSequence,
    const std::vector<ReachedTrajectoryPoint>& reached) {
  if (reached.empty()) {
    return previousSequence;
  }

  // Helix path from a reached point to one surface, folded into one turn;
  // 0 when there is none. prepareDirectNavigationSequence on a one-surface
  // sequence is exactly that, with exactly its geometry (stereo crossing
  // fixed point, concentric cylinders by radius), just evaluated from the
  // fitted state instead of the seed.
  auto helixPathFrom = [&](const ReachedTrajectoryPoint& from,
                           const Acts::Surface* srf) {
    std::vector<const Acts::Surface*> one{srf};
    return prepareDirectNavigationSequence(geoContext, from.position,
                                           from.direction, one, {},
                                           from.signedHelixRadius);
  };
  auto turnLengthAt = [](const ReachedTrajectoryPoint& from) {
    const double sinTheta = from.direction.head<2>().norm();
    if (!(std::abs(from.signedHelixRadius) > 0.) || !(sinTheta > 1e-6)) {
      return std::numeric_limits<double>::infinity();
    }
    return 2. * std::numbers::pi * std::abs(from.signedHelixRadius) / sinTheta;
  };

  // Reached indices per surface, in trajectory order: a material surface can
  // sit in the sequence once per turn, and its n-th occurrence there is its
  // n-th crossing.
  std::unordered_map<const Acts::Surface*, std::deque<std::size_t>> crossings;
  for (std::size_t j = 0; j < reached.size(); ++j) {
    crossings[reached[j].surface].push_back(j);
  }

  constexpr double kKeepOrder = 1e-6;   // [mm], keeps ties in the old order
  std::vector<std::pair<double, std::size_t>> keyed;
  keyed.reserve(previousSequence.size());
  std::vector<std::size_t> beforeFirst;
  std::ptrdiff_t anchor = -1;
  // Turn unwrapping along a run of unreached surfaces behind one anchor, as in
  // prepareDirectNavigationSequence.
  double runOffset = 0.;
  double runPrevFolded = 0.;
  bool runStarted = false;
  std::size_t tieBreak = 0;

  for (std::size_t i = 0; i < previousSequence.size(); ++i) {
    const Acts::Surface* srf = previousSequence[i];
    auto it = crossings.find(srf);
    if (it != crossings.end() && !it->second.empty()) {
      anchor = static_cast<std::ptrdiff_t>(it->second.front());
      it->second.pop_front();
      keyed.emplace_back(reached[anchor].pathLength, i);
      runStarted = false;
      continue;
    }
    if (anchor < 0) {
      // Unreached and before anything reached: nothing fitted to place it by.
      beforeFirst.push_back(i);
      continue;
    }
    const ReachedTrajectoryPoint& from = reached[anchor];
    const double folded = helixPathFrom(from, srf);
    const double turn = turnLengthAt(from);
    double key = 0.;
    if (!(folded > 0.) || !std::isfinite(folded)) {
      // The fitted helix does not reach it: keep it right behind its anchor,
      // in the old order, rather than guess.
      key = from.pathLength + kKeepOrder * static_cast<double>(++tieBreak);
    } else {
      if (!runStarted) {
        // The first unreached surface behind an anchor was usually skipped
        // because it lay slightly BEHIND the navigator; folded, that reads as
        // almost a full turn ahead. Take it as behind the anchor instead, so
        // it swaps in front of it, which is the order the track flew.
        runOffset = (std::isfinite(turn) && folded > 0.5 * turn) ? -turn : 0.;
        runStarted = true;
      } else if (std::isfinite(turn) && folded < runPrevFolded - 0.5 * turn) {
        runOffset += turn;
      }
      runPrevFolded = folded;
      key = from.pathLength + folded + runOffset;
    }
    keyed.emplace_back(key, i);
  }

  // Unreached surfaces before the first reached one keep their order, just in
  // front of it (the next pass starts on a later surface anyway).
  const double front = reached.front().pathLength;
  for (std::size_t k = 0; k < beforeFirst.size(); ++k) {
    keyed.emplace_back(
        front - kKeepOrder * static_cast<double>(beforeFirst.size() - k),
        beforeFirst[k]);
  }

  std::stable_sort(keyed.begin(), keyed.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<const Acts::Surface*> resorted;
  resorted.reserve(keyed.size());
  for (const auto& [key, i] : keyed) {
    (void)key;
    // Same de-duplication as prepareDirectNavigationSequence: two crossings
    // of one surface with nothing in between are one.
    if (resorted.empty() || resorted.back() != previousSequence[i]) {
      resorted.push_back(previousSequence[i]);
    }
  }
  return resorted;
}

}  // namespace ActsExamples

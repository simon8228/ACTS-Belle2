// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Direction.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Propagator/NavigationTarget.hpp"
#include "Acts/Propagator/NavigatorOptions.hpp"
#include "Acts/Propagator/NavigatorStatistics.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/LineBounds.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/Intersection.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Acts {

namespace detail {

/// Look-ahead window for the DirectNavigator's sequence reorder, read once from
/// ACTS_DN_LOOKAHEAD. 0 (the default) disables it, leaving the navigator
/// byte-for-byte as it was.
///
/// The navigator walks its sequence with a monotonically advancing index and
/// drops any surface it cannot intersect ahead of the current position
/// (`nearLimit`, -100 um). There is no way back, so a sequence that is locally
/// out of order loses hits silently: the pair is visited in the wrong order,
/// the second one is then behind, and it is skipped for good. The fitters never
/// see it -- the surface is not a hole and not an outlier, it simply never
/// becomes a track state.
///
/// That happens because the sequence is sorted analytically, before the fit,
/// from the seed helix in a uniform field with no energy loss
/// (ActsExamples::prepareDirectNavigationSequence), while the propagation
/// follows the real trajectory. Measured on a 400-event Belle II CDC DAF fit,
/// hits lost this way are enriched 9x in truth-order inversion against a
/// sequence neighbour (18.5% against 2.0% for hits that were processed), and
/// the median arc-length gap between neighbouring straws is only ~19 mm, so a
/// small seed error is enough to flip a pair.
///
/// The window closes exactly that gap: rather than trusting slot i, intersect
/// the next `1 + lookahead` surfaces from where the track ACTUALLY is and take
/// the one it reaches first, swapping it into slot i. On an already-correct
/// sequence the nearest surface IS the next one, so nothing moves. This is a
/// cheap, local version of what GenFit's KalmanFitterRefTrack gets by sorting
/// on the propagated path length and re-preparing until the order stops
/// changing (KalmanFitterRefTrack::processTrackWithRep).
///
/// Use 1. Measured on the same 400-event fit, of the 713 hits still lost this
/// way once loop protection is off, a window of 1 would have reached 210
/// (29.5%) first -- and widening it barely moves that: 29.7% at 2, 29.9% at 3,
/// 30.4% at 5. The recoverable population is adjacent pairs, so a wider window
/// buys nothing while inviting a mis-rank, the intersections being taken with
/// `BoundaryTolerance::Infinite()` (a distant surface can report a short path
/// to a point nowhere near its bounds).
///
/// The other ~70% are NOT a local ordering problem and this knob cannot reach
/// them; they are still unexplained.
///
/// Default 1 since 2026-10-02 (with the helix ranking and the 5 mm sensitive
/// near limit below: the navigation study's best combination, FIT_CHANGES
/// N4/N9/N10). ACTS_DN_LOOKAHEAD=0 restores the plain sequence.
///
/// @note A research knob, matching how the Belle II fork A/Bs the rest of the
///       direct-navigation chain. It belongs in DirectNavigator::Options before
///       this goes anywhere near upstream.
/// @return The configured look-ahead window
/// Per-thread override of directNavigatorLookahead(), set around a single fit
/// by the DAF look-ahead fallback (DafTrackFittingAlgorithm,
/// ACTS_DAF_LOOKAHEAD_FALLBACK). Unset, the environment value applies.
inline thread_local std::optional<std::size_t> directNavigatorLookaheadOverride;

inline std::size_t directNavigatorLookahead() {
  if (directNavigatorLookaheadOverride.has_value()) {
    return *directNavigatorLookaheadOverride;
  }
  static const std::size_t value = [] {
    const char* env = std::getenv("ACTS_DN_LOOKAHEAD");
    return env == nullptr
               ? std::size_t{1}
               : static_cast<std::size_t>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// ACTS_DN_NEAR_LIMIT_MM=X: accept a SENSITIVE surface whose intersection lies
/// up to X mm behind the current position, instead of Options::nearLimit
/// (-0.1 mm). The stepper takes a target behind it (its own near limit is
/// -inf), so the track steps back to the straw it just passed instead of
/// dropping it for good. The test is on the geometry id's sensitive field,
/// and the CDC material cylinders carry one too, so in the CDC they get the
/// same limit. That is kept on purpose: a layer cylinder the look-ahead jumped
/// past is stepped back to and its material applied. Testing by surface type
/// instead (tried 2026-10-15) skipped ~30000 more cylinders in 3000 CATFinder
/// events, and the missing material cost more hits than it saved. The one
/// harmful case, stepping back from a pass's start straw onto the cylinder
/// just behind it and then re-filtering the straw, is caught by the DAF actor
/// instead (DafFitterResult::filteredMeasurementSurfaces). Default 5 mm
/// since 2026-10-02 (FIT_CHANGES N9); X <= 0 turns it off. Returns the near
/// limit to use (negative), or 0 when off.
inline double directNavigatorSensitiveNearLimit() {
  static const double value = [] {
    const char* env = std::getenv("ACTS_DN_NEAR_LIMIT_MM");
    const double mm = env == nullptr ? 5. : std::strtod(env, nullptr);
    return mm > 0. ? -mm * UnitConstants::mm : 0.;
  }();
  return value;
}

/// ACTS_DN_HELIX_RANK=1: rank the look-ahead candidates (ACTS_DN_LOOKAHEAD)
/// by a curvature-corrected path length instead of the straight-line one,
/// using the q/p * B the fitter's actor stores in State::qopField. The
/// straight-line distance to a straw mis-ranks neighbours on curling tracks
/// and at shallow crossing angles. No effect without a look-ahead window or
/// without an actor that fills qopField (only the DAF's does). Default on
/// since 2026-10-02 (FIT_CHANGES N10); ACTS_DN_HELIX_RANK=0 turns it off.
inline bool directNavigatorHelixRank() {
  static const bool value = [] {
    const char* env = std::getenv("ACTS_DN_HELIX_RANK");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return value;
}

/// ACTS_DN_HELIX_HOLD=N (0 or unset = off): when neither the next surface
/// nor the look-ahead window has an intersection ahead on the current
/// STRAIGHT line, but the local helix (position, direction, qopField) passes
/// the straw within its cell further on, keep that surface as the next one to
/// try and step on without a target -- at most N times in a row per surface --
/// instead of skipping it for good.
///
/// Why: a low-pT track curling inside the CDC turns round. Near the turning
/// point the next straws, on the returning arm, lie behind the current
/// straight direction although the helix reaches them a few cm later. They
/// were skipped one after the other until the sequence ran out, the navigation
/// broke, and the pass coasted to its path limit: phase3s2 B events, forward
/// 20-60 deg below 0.3 GeV, 6.6% of hits never reached and 92% of those in a
/// contiguous TAIL at a median 0.72 helix turns. GenFit extrapolates the helix
/// to every wire and has no such test.
///
/// Needs qopField (filled by the fitter actors with the default helix rank).
inline unsigned int directNavigatorHelixHold() {
  static const unsigned int value = [] {
    const char* env = std::getenv("ACTS_DN_HELIX_HOLD");
    return env == nullptr ? 0u
                          : static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// ACTS_DN_DROP_UNREACHABLE=N (0 or unset = off): every N steps spent on one
/// target, check that the local helix (position, direction, qopField) can
/// still reach it -- a straw within its cell inside one turn, a concentric
/// material cylinder inside the helix's radius range -- and drop it if not.
/// Material cylinders the helix cannot reach are also skipped when the next
/// target is chosen.
///
/// Why: the navigator never gives up on a target by itself. A target the
/// track can no longer reach (a material cylinder beyond the radius a curling
/// track turns at, a straw the fitted track misses) stays "reachable" on the
/// straight line, so the pass flies on to its path limit and every straw
/// after it in the sequence is lost. On 200 CATFinder events (proto track 10
/// of each event, look-ahead 1) that was 2595 of 3282 lost straws, against 687
/// dropped as behind (2026-10-08).
inline unsigned int directNavigatorDropUnreachable() {
  static const unsigned int value = [] {
    const char* env = std::getenv("ACTS_DN_DROP_UNREACHABLE");
    return env == nullptr ? 0u
                          : static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
  }();
  return value;
}

/// Counters for ACTS_DN_DROP_UNREACHABLE, printed once at exit.
struct DirectNavigatorDropCounters {
  std::atomic<long> straws{0};
  std::atomic<long> material{0};
  std::atomic<long> materialSkipped{0};
  ~DirectNavigatorDropCounters() {
    if (straws + material + materialSkipped > 0) {
      std::fprintf(stderr,
                   "DirectNavigator (ACTS_DN_DROP_UNREACHABLE): dropped %ld "
                   "straw and %ld material targets the local helix could not "
                   "reach; skipped %ld unreachable material surfaces\n",
                   straws.load(), material.load(), materialSkipped.load());
    }
  }
};
inline DirectNavigatorDropCounters& directNavigatorDropCounters() {
  static DirectNavigatorDropCounters counters;
  return counters;
}

}  // namespace detail

/// @brief A fully guided navigator
///
/// This is a fully guided navigator that progresses through a provided sequence
/// of surfaces.
///
/// This can either be used as a validation tool, for truth tracking, or track
/// refitting.
///
class DirectNavigator {
 public:
  /// @brief The nested configuration struct
  struct Config {};

  /// @brief The nested options struct
  struct Options : public NavigatorPlainOptions {
    /// Constructor from geometry context
    /// @param gctx The geometry context
    explicit Options(const GeometryContext& gctx)
        : NavigatorPlainOptions(gctx) {}

    /// The surface tolerance
    double surfaceTolerance = s_onSurfaceTolerance;

    // TODO https://github.com/acts-project/acts/issues/2738
    /// Distance limit to discard intersections "behind us"
    /// @note this is only necessary because some surfaces have more than one
    ///       intersection
    double nearLimit = -100 * UnitConstants::um;

    /// The far limit to resolve surfaces
    double farLimit = std::numeric_limits<double>::max();

    /// Record every surface given up on in State::skips (the DAF's
    /// navigation trace sets this; off by default)
    bool recordSkips = false;

    /// Set the plain navigator options
    /// @param options The plain navigator options to copy
    void setPlainOptions(const NavigatorPlainOptions& options) {
      static_cast<NavigatorPlainOptions&>(*this) = options;
    }
  };

  /// @brief Nested State struct
  ///
  /// It acts as an internal state which is created for every
  /// propagation/extrapolation step and keep thread-local navigation
  /// information
  struct State {
    /// Constructor from options
    /// @param options_ The navigator options
    explicit State(const Options& options_) : options(options_) {}

    /// Configuration options for the direct navigator
    Options options;

    /// Propagation direction (forward or backward)
    Direction direction = Direction::Forward();

    /// Index of the next surface to try
    /// @note -1 means before the first surface in the sequence and size()
    ///       means after the last surface in the sequence
    std::int32_t surfaceIndex = -1;

    /// Navigation state - external interface: the current surface
    const Surface* currentSurface = nullptr;

    /// Navigation state - external interface: a break has been detected
    bool navigationBreak = false;

    /// q/p times the field at the current position, d(direction)/ds =
    /// direction x qopField. Filled by a fitter actor while
    /// ACTS_DN_HELIX_RANK is on (see detail::directNavigatorHelixRank);
    /// hasQopField says whether it is current.
    Vector3 qopField = Vector3::Zero();
    bool hasQopField = false;

    /// ACTS_DN_HELIX_HOLD: the surface index currently held, and how many
    /// consecutive steps it has been held for.
    std::int32_t heldIndex = -2;
    unsigned int holdCount = 0;

    /// ACTS_DN_DROP_UNREACHABLE: the surface index of the current target and
    /// the steps spent on it.
    std::int32_t pursuedIndex = -2;
    unsigned int pursuedSteps = 0;

    /// Why the navigator gave up on a surface of the sequence.
    enum class SkipReason : std::uint8_t {
      /// Its nearest straight-line intersection lies behind the near limit
      Behind = 1,
      /// No valid straight-line intersection at all
      NoIntersection = 2,
      /// ACTS_DN_DROP_UNREACHABLE dropped it as the current target
      DroppedUnreachable = 3,
      /// ACTS_DN_DROP_UNREACHABLE skipped an unreachable material cylinder
      MaterialUnreachable = 4,
    };
    /// One surface given up on. The index never comes back, so every entry is
    /// a surface this propagation can no longer visit.
    struct Skip {
      std::int32_t index = -1;
      const Surface* surface = nullptr;
      SkipReason reason = SkipReason::Behind;
      /// Signed path to the nearest straight-line intersection, NaN if none
      double pathToSurface = std::numeric_limits<double>::quiet_NaN();
      /// Track position when it was given up on
      double positionR = 0.;
      double positionZ = 0.;
      /// The surface in the slot before it: for the first skip after a
      /// target was reached, the surface just reached
      const Surface* previous = nullptr;
    };
    /// Filled only while Options::recordSkips is set; empty otherwise.
    std::vector<Skip> skips;

    /// Navigation statistics
    NavigatorStatistics statistics;

    /// Get the current navigation surface
    /// @return Reference to the surface at the current surface index
    const Surface& navSurface() const {
      return *options.externalSurfaces.at(surfaceIndex);
    }

    /// Move to the next surface in the sequence
    /// Increments or decrements surface index based on propagation direction
    void nextSurface() {
      if (direction == Direction::Forward()) {
        ++surfaceIndex;
      } else {
        --surfaceIndex;
      }
    }

    /// Check if we have reached the end of the surface sequence
    /// @return True if no more surfaces remain in the propagation direction
    bool endOfSurfaces() const {
      if (direction == Direction::Forward()) {
        return surfaceIndex >=
               static_cast<int>(options.externalSurfaces.size());
      }
      return surfaceIndex < 0;
    }

    /// Get the number of surfaces remaining in the sequence
    /// @return Number of surfaces left to process in the propagation direction
    int remainingSurfaces() const {
      if (direction == Direction::Forward()) {
        return options.externalSurfaces.size() - surfaceIndex;
      }
      return surfaceIndex + 1;
    }

    /// Reset the surface index to the initial position
    /// Sets index to before first surface (forward) or after last surface
    /// (backward)
    void resetSurfaceIndex() {
      surfaceIndex = direction == Direction::Forward()
                         ? -1
                         : static_cast<int>(options.externalSurfaces.size());
    }
  };

  /// Constructor with optional logger
  /// @param _logger Logger instance for navigation messages
  explicit DirectNavigator(std::unique_ptr<const Logger> _logger =
                               getDefaultLogger("DirectNavigator",
                                                Logging::INFO))
      : m_logger{std::move(_logger)} {}

  /// Create a new navigation state from options
  /// @param options The navigator options
  /// @return New state initialized with the provided options
  State makeState(const Options& options) const {
    State state(options);
    return state;
  }

  /// Get the current surface from the navigation state
  /// @param state The navigation state
  /// @return Pointer to current surface, nullptr if none
  const Surface* currentSurface(const State& state) const {
    return state.currentSurface;
  }

  /// Get the current tracking volume (not used by DirectNavigator)
  /// @param state The navigation state (unused)
  /// @return Always returns nullptr as DirectNavigator doesn't use volumes
  const TrackingVolume* currentVolume(const State& state) const {
    static_cast<void>(state);
    return nullptr;
  }

  /// Get the current volume material (not used by DirectNavigator)
  /// @param state The navigation state (unused)
  /// @return Always returns nullptr as DirectNavigator doesn't use volume material
  const IVolumeMaterial* currentVolumeMaterial(const State& state) const {
    static_cast<void>(state);
    return nullptr;
  }

  /// Get the start surface from the navigation state
  /// @param state The navigation state
  /// @return Pointer to start surface, nullptr if none
  const Surface* startSurface(const State& state) const {
    return state.options.startSurface;
  }

  /// Get the target surface from the navigation state
  /// @param state The navigation state
  /// @return Pointer to target surface, nullptr if none
  const Surface* targetSurface(const State& state) const {
    return state.options.targetSurface;
  }

  /// Check if the end of world has been reached (not applicable for
  /// DirectNavigator)
  /// @param state The navigation state (unused)
  /// @return Always returns false as DirectNavigator operates on a surface sequence
  bool endOfWorldReached(State& state) const {
    static_cast<void>(state);
    return false;
  }

  /// Check if navigation should break/stop
  /// @param state The navigation state
  /// @return True if navigation should break, false otherwise
  bool navigationBreak(const State& state) const {
    return state.navigationBreak;
  }

  /// @brief Initialize the navigator
  ///
  /// This function initializes the navigator for a new propagation.
  ///
  /// @param state The navigation state
  /// @param position The start position
  /// @param direction The start direction
  /// @param propagationDirection The propagation direction
  /// @return Always returns success result as DirectNavigator initialization cannot fail
  [[nodiscard]] Result<void> initialize(State& state, const Vector3& position,
                                        const Vector3& direction,
                                        Direction propagationDirection) const {
    static_cast<void>(position);
    static_cast<void>(direction);

    ACTS_VERBOSE("Initialize. Surface sequence for navigation:");
    for (const Surface* surface : state.options.externalSurfaces) {
      ACTS_VERBOSE(surface->geometryId()
                   << " - "
                   << surface->center(state.options.geoContext).transpose());
    }

    state.direction = propagationDirection;
    ACTS_VERBOSE("Navigation direction is " << propagationDirection);

    // We set the current surface to the start surface
    state.currentSurface = state.options.startSurface;
    if (state.currentSurface != nullptr) {
      ACTS_VERBOSE("Current surface set to start surface "
                   << state.currentSurface->geometryId());
    } else {
      ACTS_VERBOSE("Current surface set to nullptr");
    }

    // Find initial index.
    auto found = std::ranges::find(state.options.externalSurfaces,
                                   state.options.startSurface);

    if (found != state.options.externalSurfaces.end()) {
      // The index should be the index before the start surface, depending on
      // the direction
      state.surfaceIndex =
          std::distance(state.options.externalSurfaces.begin(), found);
      state.surfaceIndex += state.direction == Direction::Backward() ? 1 : -1;
    } else {
      ACTS_DEBUG(
          "Did not find the start surface in the sequence. Assuming it is not "
          "part of the sequence. Trusting the correctness of the input "
          "sequence. Resetting the surface index.");
      state.resetSurfaceIndex();
    }

    state.navigationBreak = false;

    return Result<void>::success();
  }

  /// @brief Get the next target surface
  ///
  /// This function gets the next target surface for the propagation. For
  /// the direct navigator this is always the next surface in the sequence.
  ///
  /// @param state The navigation state
  /// @param position The current position
  /// @param direction The current direction
  ///
  /// @return The next target surface
  NavigationTarget nextTarget(State& state, const Vector3& position,
                              const Vector3& direction) const {
    // Navigator target always resets the current surface
    state.currentSurface = nullptr;

    if (state.navigationBreak) {
      return NavigationTarget::None();
    }

    ACTS_VERBOSE("DirectNavigator::nextTarget");

    // Move the sequence to the next surface
    state.nextSurface();

    // Surfaces considered per step: just the next one unless a look-ahead
    // window is configured, in which case the nearest of the next few wins and
    // is swapped into place. See detail::directNavigatorLookahead.
    const std::size_t lookahead = detail::directNavigatorLookahead();

    while (!state.endOfSurfaces()) {
      ACTS_VERBOSE("Next surface candidate is "
                   << state.navSurface().geometryId() << ". "
                   << state.remainingSurfaces() << " out of "
                   << state.options.externalSurfaces.size()
                   << " surfaces remain to try.");

      // Establish & update the surface status
      // TODO we do not know the intersection index - passing the closer one
      const double farLimit = std::numeric_limits<double>::max();
      const std::int32_t step =
          state.direction == Direction::Forward() ? 1 : -1;
      const std::int32_t nSurfaces =
          static_cast<std::int32_t>(state.options.externalSurfaces.size());

      NavigationTarget best = NavigationTarget::None();
      std::int32_t bestIndex = state.surfaceIndex;
      double bestRank = std::numeric_limits<double>::max();

      for (std::size_t ahead = 0; ahead <= lookahead; ++ahead) {
        const std::int32_t index =
            state.surfaceIndex + step * static_cast<std::int32_t>(ahead);
        if (index < 0 || index >= nSurfaces) {
          break;
        }
        const Surface& candidateSurface = *state.options.externalSurfaces[index];
        if (detail::directNavigatorDropUnreachable() > 0 && state.hasQopField &&
            candidateSurface.type() == Surface::Cylinder &&
            !cylinderReachable(candidateSurface, position, direction,
                               state.qopField)) {
          ++detail::directNavigatorDropCounters().materialSkipped;
          continue;
        }
        // A straw just passed is still a valid target when
        // ACTS_DN_NEAR_LIMIT_MM is set. By id, so the CDC material cylinders
        // (which carry a sensitive id) get it too -- see
        // directNavigatorSensitiveNearLimit for why that stays.
        double nearLimit = state.options.nearLimit;
        const double sensitiveNear = detail::directNavigatorSensitiveNearLimit();
        if (sensitiveNear < 0. && candidateSurface.geometryId().sensitive() != 0) {
          nearLimit = std::min(nearLimit, sensitiveNear);
        }
        const NavigationTarget candidate = chooseIntersection(
            state.options.geoContext, candidateSurface, position, direction,
            BoundaryTolerance::Infinite(), nearLimit, farLimit,
            state.options.surfaceTolerance);
        if (!candidate.isValid()) {
          continue;
        }
        // Reached first wins. With lookahead == 0 this runs once and the
        // behaviour is identical to taking state.navSurface() directly.
        const double rank =
            (lookahead > 0 && detail::directNavigatorHelixRank() &&
             state.hasQopField && state.direction == Direction::Forward())
                ? helixPathLength(state.options.geoContext, candidateSurface,
                                  position, direction, state.qopField,
                                  candidate.pathLength(),
                                  state.options.surfaceTolerance)
                : candidate.pathLength();
        if (!best.isValid() || rank < bestRank) {
          best = candidate;
          bestIndex = index;
          bestRank = rank;
        }
      }

      if (best.isValid()) {
        if (bestIndex != state.surfaceIndex) {
          // Both slots are ahead of the walk, so reordering them cannot
          // disturb anything already visited; navSurface() and
          // handleSurfaceReached stay consistent because the chosen surface
          // now sits at the current index.
          ACTS_VERBOSE(
              "Reordering: "
              << state.options.externalSurfaces[bestIndex]->geometryId()
              << " is reached before "
              << state.options.externalSurfaces[state.surfaceIndex]->geometryId()
              << ", moving it up " << (bestIndex - state.surfaceIndex)
              << " slot(s).");
          std::swap(state.options.externalSurfaces[state.surfaceIndex],
                    state.options.externalSurfaces[bestIndex]);
        }
        return best;
      }

      // ACTS_DN_HELIX_HOLD: not ahead on the straight line, but on the helix?
      if (const unsigned int maxHolds = detail::directNavigatorHelixHold();
          maxHolds > 0 && state.hasQopField &&
          state.direction == Direction::Forward()) {
        if (state.heldIndex != state.surfaceIndex) {
          state.heldIndex = state.surfaceIndex;
          state.holdCount = 0;
        }
        if (state.holdCount < maxHolds &&
            helixReachesStraw(state.options.geoContext, state.navSurface(),
                              position, direction, state.qopField)) {
          if (state.holdCount == 0) {
            static std::atomic<long> nHeld{0};
            if ((nHeld.fetch_add(1) + 1) % 10000 == 0) {
              std::fprintf(stderr,
                           "DirectNavigator: %ld straws held for the helix "
                           "(ACTS_DN_HELIX_HOLD)\n",
                           nHeld.load());
            }
          }
          ++state.holdCount;
          ACTS_VERBOSE("Holding " << state.navSurface().geometryId()
                                  << ": behind the straight line, ahead on "
                                     "the helix (hold "
                                  << state.holdCount << ")");
          // Undo nextSurface(): the next call tries this surface again,
          // from wherever the track has curved to by then.
          state.surfaceIndex -= step;
          return NavigationTarget::None();
        }
      }

      ACTS_VERBOSE("No valid intersection found with surface "
                   << state.navSurface().geometryId()
                   << ", trying next surface.");
      if (state.options.recordSkips) {
        recordSkip(state, position, direction);
      }
      state.nextSurface();
    }

    ACTS_VERBOSE("End of surfaces reached, navigation break.");
    state.navigationBreak = true;
    return NavigationTarget::None();
  }

  /// @brief Check if the current target is still valid
  ///
  /// This function checks if the target is valid. For the direct navigator this
  /// is always true.
  ///
  /// @param state The navigation state
  /// @param position The current position
  /// @param direction The current direction
  ///
  /// @return True if the target is valid
  bool checkTargetValid(State& state, const Vector3& position,
                        const Vector3& direction) const {
    const unsigned int every = detail::directNavigatorDropUnreachable();
    if (every == 0 || !state.hasQopField ||
        state.direction != Direction::Forward() || state.surfaceIndex < 0 ||
        state.endOfSurfaces()) {
      return true;
    }
    if (state.pursuedIndex != state.surfaceIndex) {
      state.pursuedIndex = state.surfaceIndex;
      state.pursuedSteps = 0;
    }
    if (++state.pursuedSteps % every != 0) {
      return true;
    }
    const Surface& target = state.navSurface();
    // By type: the CDC material cylinders carry a sensitive id too.
    const bool sensitive = target.type() == Surface::Straw;
    const bool reachable =
        sensitive ? helixPassesStraw(state.options.geoContext, target, position,
                                     direction, state.qopField)
                  : cylinderReachable(target, position, direction,
                                      state.qopField);
    if (reachable) {
      return true;
    }
    if (sensitive) {
      ++detail::directNavigatorDropCounters().straws;
    } else {
      ++detail::directNavigatorDropCounters().material;
    }
    ACTS_VERBOSE("Dropping target " << target.geometryId()
                                    << ": the local helix no longer reaches it "
                                       "(ACTS_DN_DROP_UNREACHABLE)");
    if (state.options.recordSkips) {
      State::Skip skip;
      skip.index = state.surfaceIndex;
      skip.surface = &target;
      skip.reason = State::SkipReason::DroppedUnreachable;
      skip.positionR = std::hypot(position.x(), position.y());
      skip.positionZ = position.z();
      state.skips.push_back(skip);
    }
    // The next nextTarget() call advances past it.
    return false;
  }

  /// @brief Handle the surface reached
  ///
  /// This function handles the surface reached. For the direct navigator this
  /// effectively sets the current surface to the reached surface.
  ///
  /// @param state The navigation state
  /// @param position The current position
  /// @param direction The current direction
  /// @param surface The surface reached
  void handleSurfaceReached(State& state, const Vector3& position,
                            const Vector3& direction,
                            const Surface& surface) const {
    static_cast<void>(position);
    static_cast<void>(direction);
    static_cast<void>(surface);

    if (state.navigationBreak) {
      return;
    }

    ACTS_VERBOSE("DirectNavigator::handleSurfaceReached");

    // Set the current surface
    state.currentSurface = &state.navSurface();
    ACTS_VERBOSE("Current surface set to "
                 << state.currentSurface->geometryId());
  }

 private:
  /// Options::recordSkips: log the surface at the current index, which
  /// nextTarget() is about to give up on, with why. The reason is worked out
  /// again here rather than carried out of the search, so the search itself is
  /// untouched: an unreachable material cylinder (ACTS_DN_DROP_UNREACHABLE),
  /// else the nearest valid straight-line intersection -- which the search
  /// rejected, so it lies behind the near limit -- else none at all.
  void recordSkip(State& state, const Vector3& position,
                  const Vector3& direction) const {
    const Surface& surface = state.navSurface();
    State::Skip skip;
    skip.index = state.surfaceIndex;
    skip.surface = &surface;
    skip.positionR = std::hypot(position.x(), position.y());
    skip.positionZ = position.z();
    {
      const std::int32_t step =
          state.direction == Direction::Forward() ? 1 : -1;
      const std::int32_t prev = state.surfaceIndex - step;
      if (prev >= 0 &&
          prev < static_cast<std::int32_t>(state.options.externalSurfaces.size())) {
        skip.previous = state.options.externalSurfaces[prev];
      }
    }
    skip.reason = State::SkipReason::NoIntersection;
    if (detail::directNavigatorDropUnreachable() > 0 && state.hasQopField &&
        surface.type() == Surface::Cylinder &&
        !cylinderReachable(surface, position, direction, state.qopField)) {
      skip.reason = State::SkipReason::MaterialUnreachable;
    } else {
      auto intersections = surface.intersect(
          state.options.geoContext, position, direction,
          BoundaryTolerance::Infinite(), state.options.surfaceTolerance);
      for (const auto& intersection : intersections) {
        if (intersection.isValid() &&
            (std::isnan(skip.pathToSurface) ||
             std::abs(intersection.pathLength()) <
                 std::abs(skip.pathToSurface))) {
          skip.pathToSurface = intersection.pathLength();
          skip.reason = State::SkipReason::Behind;
        }
      }
    }
    state.skips.push_back(skip);
  }

  /// ACTS_DN_HELIX_HOLD: does the local helix from (position, direction),
  /// d(direction)/ds = direction x qopField, pass this straw within its cell
  /// ahead of here? Exact helix in the local field, sampled every 2 mm up to
  /// half a turn or 1.5 m; "within its cell" is 3x the straw's bound radius,
  /// generous on purpose, since this only decides hold-or-skip.
  static bool helixReachesStraw(const GeometryContext& gctx,
                                const Surface& surface,
                                const Vector3& position,
                                const Vector3& direction,
                                const Vector3& qopField) {
    if (surface.type() != Surface::Straw) {
      return false;
    }
    const auto* bounds = dynamic_cast<const LineBounds*>(&surface.bounds());
    const double omega = qopField.norm();
    if (bounds == nullptr || !(omega > 0.)) {
      return false;
    }
    const double reach = 3. * bounds->get(LineBounds::eR);
    const Vector3 b = qopField / omega;
    const Vector3 dPar = direction.dot(b) * b;
    const Vector3 dPerp = direction - dPar;
    const Vector3 dCross = dPerp.cross(b);
    const Vector3 center = surface.localToGlobalTransform(gctx).translation();
    const Vector3 axis = surface.localToGlobalTransform(gctx).rotation().col(2);
    auto distance = [&](double s) {
      const Vector3 x = position + dPar * s + dPerp * (std::sin(omega * s) / omega) +
                        dCross * ((1. - std::cos(omega * s)) / omega);
      Vector3 r = x - center;
      r -= r.dot(axis) * axis;
      return r.norm();
    };
    const double step = 2. * UnitConstants::mm;
    const double sMax = std::min(M_PI / omega, 1.5 * UnitConstants::m);
    double best = distance(0.);
    double bestS = 0.;
    for (double s = step; s <= sMax; s += step) {
      const double dist = distance(s);
      if (dist < best) {
        best = dist;
        bestS = s;
      }
    }
    return bestS > step && best <= reach;
  }

  /// ACTS_DN_DROP_UNREACHABLE: does the local helix pass a straw within
  /// 3x the bound radius + 30 mm at any point from here to one full turn (at
  /// most 3 m) ahead? Unlike helixReachesStraw this counts the current point
  /// too, so a straw the track is just reaching is not dropped. The 30 mm is
  /// for the seed: CATFinder's misses its first straw by more than a cell
  /// (with 3x alone the first straw of a track was dropped), and the fit only
  /// pulls the track onto it once there. Only targets the track cannot get
  /// near at all are to be dropped.
  static bool helixPassesStraw(const GeometryContext& gctx,
                               const Surface& surface, const Vector3& position,
                               const Vector3& direction,
                               const Vector3& qopField) {
    if (surface.type() != Surface::Straw) {
      return true;
    }
    const auto* bounds = dynamic_cast<const LineBounds*>(&surface.bounds());
    const double omega = qopField.norm();
    if (bounds == nullptr || !(omega > 0.)) {
      return true;
    }
    const double reach =
        3. * bounds->get(LineBounds::eR) + 30. * UnitConstants::mm;
    const Vector3 b = qopField / omega;
    const Vector3 dPar = direction.dot(b) * b;
    const Vector3 dPerp = direction - dPar;
    const Vector3 dCross = dPerp.cross(b);
    const Vector3 center = surface.localToGlobalTransform(gctx).translation();
    const Vector3 axis = surface.localToGlobalTransform(gctx).rotation().col(2);
    const double step = 3. * UnitConstants::mm;
    const double sMax = std::min(2. * M_PI / omega, 3. * UnitConstants::m);
    for (double s = 0.; s <= sMax; s += step) {
      const Vector3 x = position + dPar * s + dPerp * (std::sin(omega * s) / omega) +
                        dCross * ((1. - std::cos(omega * s)) / omega);
      Vector3 r = x - center;
      r -= r.dot(axis) * axis;
      if (r.norm() <= reach) {
        return true;
      }
    }
    return false;
  }

  /// ACTS_DN_DROP_UNREACHABLE: can the local helix reach the radius of a
  /// concentric cylinder? Its transverse distance from the z axis stays in
  /// [|c| - rho, |c| + rho] for a circle of radius rho about c (field taken
  /// along z). 10 mm of slack for the field map and energy loss. Anything that
  /// is not a concentric cylinder counts as reachable.
  static bool cylinderReachable(const Surface& surface, const Vector3& position,
                                const Vector3& direction,
                                const Vector3& qopField) {
    if (surface.type() != Surface::Cylinder) {
      return true;
    }
    const auto* bounds = dynamic_cast<const CylinderBounds*>(&surface.bounds());
    const double omega = qopField.norm();
    if (bounds == nullptr || !(omega > 0.)) {
      return true;
    }
    const double radius = bounds->get(CylinderBounds::eR);
    const Vector3 b = qopField / omega;
    const Vector3 dPerp = direction - direction.dot(b) * b;
    const double sinAlpha = dPerp.norm();
    if (!(sinAlpha > 1e-6)) {
      return true;
    }
    const double rho = sinAlpha / omega;
    // d(direction)/ds = omega * direction x b points to the centre
    const Vector3 toCentre = dPerp.cross(b) / sinAlpha;
    const Vector3 c = position + rho * toCentre;
    const double cr = std::hypot(c.x(), c.y());
    const double slack = 10. * UnitConstants::mm;
    return radius <= cr + rho + slack && radius >= std::abs(cr - rho) - slack;
  }

  /// Path length to `surface` along the local helix, from the straight-line
  /// estimate `straightPath`: two fixed-point steps that re-intersect from the
  /// second-order position and direction at the current estimate
  /// (x(s) = x + s d + s^2/2 d x k, d(s) = d + s d x k, k = q/p * B). Used
  /// only to rank look-ahead candidates; the target itself is unchanged.
  static double helixPathLength(const GeometryContext& gctx,
                                const Surface& surface,
                                const Vector3& position,
                                const Vector3& direction,
                                const Vector3& qopField, double straightPath,
                                double tolerance) {
    double s = straightPath;
    const Vector3 bend = direction.cross(qopField);
    for (int iteration = 0; iteration < 2; ++iteration) {
      const Vector3 x = position + s * direction + 0.5 * s * s * bend;
      const Vector3 d = (direction + s * bend).normalized();
      auto intersections = surface.intersect(
          gctx, x, d, BoundaryTolerance::Infinite(), tolerance);
      // The correction is the solution nearest the predicted point. Taking
      // the first valid one is wrong for cylinders: their solutions are
      // sorted by signed path length, so the first is the far side behind,
      // which ranked every material cylinder ahead of the straw next to it
      // and dropped that straw (2026-10-01, FIT_CHANGES N10).
      bool found = false;
      double correction = 0.;
      for (const auto& intersection : intersections) {
        if (intersection.isValid() &&
            (!found ||
             std::abs(intersection.pathLength()) < std::abs(correction))) {
          correction = intersection.pathLength();
          found = true;
        }
      }
      if (!found) {
        return straightPath;
      }
      s += correction;
    }
    return s;
  }

  NavigationTarget chooseIntersection(
      const GeometryContext& gctx, const Surface& surface,
      const Vector3& position, const Vector3& direction,
      const BoundaryTolerance& boundaryTolerance, double nearLimit,
      double farLimit, double tolerance) const {
    auto intersections = surface.intersect(gctx, position, direction,
                                           boundaryTolerance, tolerance);

    for (auto [intersectionIndex, intersection] :
         Acts::enumerate(intersections)) {
      if (intersection.isValid() &&
          detail::checkPathLength(intersection.pathLength(), nearLimit,
                                  farLimit, logger())) {
        return NavigationTarget(intersection, intersectionIndex, surface,
                                boundaryTolerance);
      }
    }

    return NavigationTarget::None();
  }

  const Logger& logger() const { return *m_logger; }

  std::unique_ptr<const Logger> m_logger;
};

}  // namespace Acts

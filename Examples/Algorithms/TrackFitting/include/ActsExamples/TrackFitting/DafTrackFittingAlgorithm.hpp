// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/EventData/TruthMatching.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/TrackFitting/DafFitterFunction.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Acts {
class TrackingGeometry;
}

namespace ActsExamples {

/// Runs the Deterministic Annealing Filter, seeded by proto tracks but not
/// bound by their hit choices: each proto track only tells this algorithm
/// which *surfaces* the track visits (via its measurement indices' geometry
/// IDs), and for every such surface the algorithm re-gathers ALL measurements
/// on that surface from the raw measurement container -- not just the one
/// the proto track builder happened to pick -- and hands the whole group to
/// DAF as competing candidates. This is what lets DAF actually resolve
/// ambiguity (e.g. SVD sensor overlaps) that a proto track already collapsed
/// to a single choice. With inputMeasurementParticlesMap set, the group is
/// restricted to measurements truth-matched to the proto track's majority
/// particle, keeping unrelated particles' hits on the same sensor out.
class DafTrackFittingAlgorithm final : public IAlgorithm {
 public:
  struct Config {
    /// Input measurements collection.
    std::string inputMeasurements;
    /// Input proto tracks collection, i.e. groups of hit indices. Only used
    /// to determine which surfaces a track visits and as the seed; the
    /// actual per-surface candidate groups are re-derived from
    /// inputMeasurements.
    std::string inputProtoTracks;
    /// Input initial track parameter estimates for for each proto track.
    std::string inputInitialTrackParameters;
    /// (optional) Input clusters for each measurement
    std::string inputClusters;
    /// (optional) Input measurement-to-particle truth map. When set, a
    /// surface's competing-candidate group only admits measurements whose
    /// truth particle matches the proto track's majority particle, so hits
    /// from unrelated particles that happen to cross the same sensor never
    /// reach DAF. When empty, every measurement sharing the geometry ID is
    /// pooled.
    std::string inputMeasurementParticlesMap;
    /// Output fitted tracks collection.
    std::string outputTracks;
    /// Tracking geometry, used to resolve surfaces from geometry IDs.
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;
    /// Type erased DAF fitter function.
    std::shared_ptr<DafFitterFunction> fit;
    /// Pick a single track for debugging (-1 process all tracks)
    int pickTrack = -1;
    /// Type erased calibrator for the measurements
    std::shared_ptr<MeasurementCalibrator> calibrator;
    /// Forward-link all tracks after fitting, enabling inside-out track state
    /// iteration via TrackProxy::trackStates(). Off by default.
    bool linkForward = false;
    /// Maximum number of propagation steps per track.
    unsigned int maxSteps = 5000;
    /// (optional) Directory to write per-event `daf-candidates.csv` files
    /// into: one row per candidate measurement per fitted track, with the
    /// DAF's final weight/chi2 and whether the candidate won its surface.
    /// These weights are otherwise internal to the annealing loop; the file
    /// also links fitted tracks back to their proto track index. Empty
    /// disables the export.
    std::string outputCandidatesDir;
    /// (optional) Alongside `daf-candidates.csv`, also write a per-event
    /// `daf-weight-trace.csv` holding the weight of every candidate after
    /// *every* annealing pass, with the temperature that pass ran at -- the
    /// annealing history rather than just its verdict. Requires
    /// outputCandidatesDir to be set. Off by default: this is roughly
    /// (passes x candidates) rows per track, ~16k rows/event for a typical
    /// CDC schedule, so it is meant for visualising a handful of events, not
    /// for production runs.
    bool outputWeightTrace = false;
    /// (optional) Restrict the weight trace to these event numbers. Empty
    /// means every event, which is what makes the trace impractical on a
    /// production run: it is roughly (passes x candidates) rows per track, so
    /// ~16k rows/event. Naming the handful of events you actually want to
    /// visualise keeps a 10k-event fit and its per-pass plots in one pass.
    std::vector<std::size_t> weightTraceEvents;
    /// (optional) Also write a per-event `daf-linearization-trace.csv`: one
    /// row per surface per annealing pass holding the state the pass
    /// linearised about, the transport Jacobian into that surface, and the
    /// surface's measurement axis there. This is what quantifies how far a
    /// plain running-state EKF's expansion point wanders between passes --
    /// i.e. how much a fixed reference track (GenFit's KalmanFitterRefTrack)
    /// could be worth on this sample.
    ///
    /// Requires outputCandidatesDir, and is restricted to the same events as
    /// weightTraceEvents. Independent of outputWeightTrace: either one being
    /// on captures the per-pass history both read from. Off by default, and
    /// noticeably heavier than the weight trace -- ~47 numbers per row
    /// against 10 -- so pick the events rather than tracing a whole run.
    bool outputLinearizationTrace = false;
    /// KF-equivalence mode: trust the proto track's hit choices verbatim
    /// instead of re-gathering competing candidates per surface. Every
    /// surface group holds exactly the proto track's own pick (first pick
    /// wins on duplicate surfaces, mirroring KalmanFitter's try_emplace),
    /// and the truth gate is bypassed. This degenerates the DAF into a
    /// Kalman-equivalent fit over the same hits as TrackFittingAlgorithm,
    /// which is the point: it isolates fitter-core differences in
    /// KF-vs-DAF comparisons.
    bool protoTrackCandidatesOnly = false;
  };

  /// Constructor of the DAF fitting algorithm
  ///
  /// @param config is the config struct to configure the algorithm
  /// @param logger is the logging level
  explicit DafTrackFittingAlgorithm(
      Config config, std::unique_ptr<const Acts::Logger> logger = nullptr);

  /// Framework execute method of the fitting algorithm
  ///
  /// @param ctx is the algorithm context that holds event-wise information
  /// @return a process code to steer the algorithm flow
  ProcessCode execute(const AlgorithmContext& ctx) const final;

  /// Get readonly access to the config parameters
  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;

  ReadDataHandle<MeasurementContainer> m_inputMeasurements{this,
                                                           "InputMeasurements"};
  ReadDataHandle<ProtoTrackContainer> m_inputProtoTracks{this,
                                                         "InputProtoTracks"};
  ReadDataHandle<TrackParametersContainer> m_inputInitialTrackParameters{
      this, "InputInitialTrackParameters"};

  ReadDataHandle<ClusterContainer> m_inputClusters{this, "InputClusters"};

  ReadDataHandle<MeasurementParticlesMap> m_inputMeasurementParticlesMap{
      this, "InputMeasurementParticlesMap"};

  WriteDataHandle<ConstTrackContainer> m_outputTracks{this, "OutputTracks"};
};

}  // namespace ActsExamples

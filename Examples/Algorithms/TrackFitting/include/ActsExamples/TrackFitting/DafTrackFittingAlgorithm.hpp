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

#include <memory>
#include <string>

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

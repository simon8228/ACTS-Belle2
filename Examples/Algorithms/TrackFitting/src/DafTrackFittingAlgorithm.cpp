// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFitting/DafTrackFittingAlgorithm.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/EventData/BoundTrackParameters.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackProxy.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFitting/DafOptions.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/MeasurementCalibration.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Validation/TrackClassification.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ActsExamples {

DafTrackFittingAlgorithm::DafTrackFittingAlgorithm(
    Config config, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("DafTrackFittingAlgorithm", std::move(logger)),
      m_cfg(std::move(config)) {
  if (m_cfg.inputMeasurements.empty()) {
    throw std::invalid_argument("Missing input measurement collection");
  }
  if (m_cfg.inputProtoTracks.empty()) {
    throw std::invalid_argument("Missing input proto tracks collection");
  }
  if (m_cfg.inputInitialTrackParameters.empty()) {
    throw std::invalid_argument(
        "Missing input initial track parameters collection");
  }
  if (m_cfg.outputTracks.empty()) {
    throw std::invalid_argument("Missing output tracks collection");
  }
  if (!m_cfg.trackingGeometry) {
    throw std::invalid_argument("Missing tracking geometry");
  }
  if (!m_cfg.calibrator) {
    throw std::invalid_argument("Missing calibrator");
  }
  if (m_cfg.inputClusters.empty() && m_cfg.calibrator->needsClusters()) {
    throw std::invalid_argument("The configured calibrator needs clusters");
  }

  m_inputMeasurements.initialize(m_cfg.inputMeasurements);
  m_inputProtoTracks.initialize(m_cfg.inputProtoTracks);
  m_inputInitialTrackParameters.initialize(m_cfg.inputInitialTrackParameters);
  m_inputClusters.maybeInitialize(m_cfg.inputClusters);
  m_inputMeasurementParticlesMap.maybeInitialize(
      m_cfg.inputMeasurementParticlesMap);
  m_outputTracks.initialize(m_cfg.outputTracks);
}

ProcessCode DafTrackFittingAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  // Read input data
  const auto& measurements = m_inputMeasurements(ctx);
  const auto& protoTracks = m_inputProtoTracks(ctx);
  const auto& initialParameters = m_inputInitialTrackParameters(ctx);

  const ClusterContainer* clusters =
      m_inputClusters.isInitialized() ? &m_inputClusters(ctx) : nullptr;

  const MeasurementParticlesMap* measurementParticlesMap =
      m_inputMeasurementParticlesMap.isInitialized()
          ? &m_inputMeasurementParticlesMap(ctx)
          : nullptr;

  ACTS_DEBUG("Input measurements: " << measurements.size());
  ACTS_DEBUG("Input proto tracks: " << protoTracks.size());
  ACTS_DEBUG("Input initial parameters: " << initialParameters.size());
  ACTS_DEBUG("Input clusters: " << (clusters ? clusters->size() : 0));

  if (protoTracks.size() != initialParameters.size()) {
    ACTS_FATAL("Inconsistent number of proto tracks and parameters "
               << protoTracks.size() << " vs " << initialParameters.size());
    return ProcessCode::ABORT;
  }

  // Construct a perigee surface as the target surface
  auto pSurface = Acts::Surface::makeShared<Acts::PerigeeSurface>(
      Acts::Vector3{0., 0., 0.});

  MeasurementCalibratorAdapter calibrator(*(m_cfg.calibrator), measurements,
                                          clusters);

  Acts::PropagatorPlainOptions propOptions(ctx.geoContext, ctx.magFieldContext);
  propOptions.maxSteps = m_cfg.maxSteps;
  DafFitterFunction::GeneralFitterOptions options{
      ctx.geoContext,
      ctx.magFieldContext,
      ctx.calibContext,
      pSurface.get(),
      propOptions,
      false};

  auto trackContainer = std::make_shared<Acts::VectorTrackContainer>();
  auto trackStateContainer = std::make_shared<Acts::VectorMultiTrajectory>();
  TrackContainer tracks(trackContainer, trackStateContainer);

  trackContainer->reserve(protoTracks.size());
  trackStateContainer->reserve(protoTracks.size() * 30);
  tracks.addColumn<std::size_t>("protoId");

  // Perform the fit for each input track. For every surface a proto track
  // touches, re-derive the full set of competing candidates from the raw
  // measurement container instead of trusting the proto track's single
  // choice -- this is what actually gives DAF something to weigh.
  std::unordered_map<const Acts::Surface*, Acts::DafMeasurementGroup>
      measurementGroups;
  // The surfaces in proto track traversal order -- the only ordering that's
  // reliably correct for DirectNavigator, since it's derived from the actual
  // pattern-recognized hit sequence rather than any geometric property (see
  // DafFitterFunction::operator()'s docs for why layer/radius sorts fail).
  std::vector<const Acts::Surface*> orderedSurfaces;
  std::size_t nAmbiguousSurfaces = 0;
  for (std::size_t itrack = 0; itrack < protoTracks.size(); ++itrack) {
    if (m_cfg.pickTrack > -1 &&
        static_cast<std::size_t>(m_cfg.pickTrack) != itrack) {
      continue;
    }

    const auto& protoTrack = protoTracks[itrack];
    const auto& initialParams = initialParameters[itrack];

    if (protoTrack.empty()) {
      ACTS_WARNING("Empty proto track " << itrack << " found.");
      continue;
    }

    ACTS_VERBOSE("Initial 4 position: "
                 << initialParams.fourPosition(ctx.geoContext).transpose());
    ACTS_VERBOSE(
        "Initial direction: " << initialParams.direction().transpose());
    ACTS_VERBOSE("Initial momentum: " << initialParams.absoluteMomentum());

    // Truth gate: only candidates truth-matched to the proto track's
    // majority particle are admitted into a surface's competing group.
    std::optional<SimBarcode> trackParticle;
    if (measurementParticlesMap != nullptr) {
      std::vector<ParticleHitCount> particleHitCounts;
      identifyContributingParticles(*measurementParticlesMap, protoTrack,
                                    particleHitCounts);
      if (particleHitCounts.empty()) {
        ACTS_WARNING("Proto track "
                     << itrack
                     << " has no truth-matched measurements; pooling "
                        "candidates unfiltered.");
      } else {
        trackParticle = particleHitCounts.front().particleId;
      }
    }

    measurementGroups.clear();
    orderedSurfaces.clear();
    for (auto measIndex : protoTrack) {
      Acts::GeometryIdentifier geoId =
          measurements.getMeasurement(measIndex).geometryId();
      const Acts::Surface* surface = m_cfg.trackingGeometry->findSurface(geoId);
      if (surface == nullptr) {
        ACTS_WARNING("Could not resolve surface for geometry id " << geoId);
        continue;
      }
      if (measurementGroups.contains(surface)) {
        // Already gathered this surface's candidates for another
        // measurement index in the same proto track.
        continue;
      }

      Acts::DafMeasurementGroup group;
      auto [begin, end] = measurements.orderedIndices().equal_range(geoId);
      std::size_t nRejected = 0;
      for (auto it = begin; it != end; ++it) {
        if (trackParticle.has_value()) {
          auto [pBegin, pEnd] =
              measurementParticlesMap->equal_range(it->index());
          const bool sameParticle =
              std::any_of(pBegin, pEnd, [&](const auto& measParticle) {
                return measParticle.second == *trackParticle;
              });
          if (!sameParticle) {
            ++nRejected;
            continue;
          }
        }
        group.push_back(Acts::SourceLink(*it));
      }
      if (group.empty()) {
        // The proto track's own pick at this surface belongs to a different
        // particle than the track's majority particle (a contaminating hit),
        // so no candidate survives -- dropping the surface is the correct
        // outcome, not an error.
        ACTS_DEBUG("All candidates on surface "
                   << geoId << " belong to other particles; skipping surface.");
        continue;
      }
      if (nRejected > 0) {
        ACTS_DEBUG("Rejected " << nRejected
                               << " other-particle candidate(s) on surface "
                               << geoId);
      }
      if (group.size() > 1) {
        ++nAmbiguousSurfaces;
      }
      measurementGroups.emplace(surface, std::move(group));
      orderedSurfaces.push_back(surface);
    }

    if (measurementGroups.empty()) {
      ACTS_WARNING("Proto track " << itrack
                                  << " resolved to no valid surfaces.");
      continue;
    }

    ACTS_VERBOSE("Invoke DAF fitter for track "
                 << itrack << " with " << measurementGroups.size()
                 << " surface group(s)");
    auto result = (*m_cfg.fit)(measurementGroups, orderedSurfaces, initialParams,
                               options, calibrator, tracks);

    if (result.ok()) {
      auto& track = result.value();
      if (track.hasReferenceSurface()) {
        track.template component<std::size_t>("protoId") = itrack;
        ACTS_VERBOSE("Fitted parameters for track " << itrack);
        ACTS_VERBOSE("  " << track.parameters().transpose());
        ACTS_VERBOSE("Measurements: (proto track->track): "
                     << protoTrack.size() << " -> " << track.nMeasurements());
      } else {
        ACTS_VERBOSE("No fitted parameters for track " << itrack);
      }
    } else {
      const Acts::Vector3 initialDir = initialParams.direction();
      const double initialP = initialParams.absoluteMomentum();
      const double initialPt =
          initialP * std::hypot(initialDir.x(), initialDir.y());
      ACTS_WARNING("DAF fit failed for track "
                   << itrack << " with error: " << result.error() << ", "
                   << result.error().message() << ". protoTrack hits="
                   << protoTrack.size() << " surfaceGroups="
                   << measurementGroups.size() << " initial p=" << initialP
                   << " initial pT=" << initialPt);
    }
  }

  ACTS_DEBUG("Fitted tracks: " << trackContainer->size());
  ACTS_DEBUG("Surfaces with >1 candidate across all tracks: "
             << nAmbiguousSurfaces);

  if (m_cfg.linkForward) {
    for (auto track : tracks) {
      track.linkForward();
    }
  }

  if (logger().doPrint(Acts::Logging::DEBUG)) {
    std::stringstream ss;
    trackStateContainer->statistics().toStream(ss);
    ACTS_DEBUG(ss.str());
  }

  ConstTrackContainer constTracks{
      std::make_shared<Acts::ConstVectorTrackContainer>(
          std::move(*trackContainer)),
      std::make_shared<Acts::ConstVectorMultiTrajectory>(
          std::move(*trackStateContainer))};

  m_outputTracks(ctx, std::move(constTracks));
  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples

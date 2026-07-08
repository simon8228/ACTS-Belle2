// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Belle2/ProtoLayerCreator.hpp"
#include "ActsExamples/DetectorCommons/Detector.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace Acts {
class IMaterialDecorator;
}

namespace ActsExamples {

class Belle2Element;

class Belle2 : public Detector {
 public:
  struct Config {
    std::size_t buildLevel = 3;
    Acts::Logging::Level logLevel = Acts::Logging::DEBUG;
    Acts::Logging::Level surfaceLogLevel = Acts::Logging::DEBUG;
    Acts::Logging::Level layerLogLevel = Acts::Logging::DEBUG;
    Acts::Logging::Level volumeLogLevel = Acts::Logging::DEBUG;
    bool buildProto = false;
    std::shared_ptr<const Acts::IMaterialDecorator> materialDecorator;
    bool gen3 = true;
    std::optional<std::filesystem::path> graphvizFile = "./belle2_graphviz.dot";
  };

  explicit Belle2(const Config& cfg);

 protected:
  void buildTrackingGeometry(
      const B2::ProtoLayerCreator::ElementFactory& detectorElementFactory);

  Config m_cfg;
};
}  // namespace ActsExamples

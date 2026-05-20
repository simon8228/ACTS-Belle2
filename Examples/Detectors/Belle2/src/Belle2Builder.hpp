// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Units.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Belle2/MyIdentifierProvider.hpp"
#include "ActsExamples/Belle2/ProtoLayerCreator.hpp"
#include "Acts/Material/IMaterialDecorator.hpp"
#include "ActsPlugins/Root/ITGeoIdentifierProvider.hpp"

namespace Acts {
class TrackingGeometry;
}

namespace ActsExamples::B2 {

class Belle2Builder {
  public:
	static constexpr float kPixelCentralModuleT = 0.15f;
	static constexpr float kPixelEndcapModuleT = 0.15f;

	static constexpr float kShortStripCentralModuleT = 0.25f;
	static constexpr float kShortStripEndcapModuleT = 0.25f;

	static constexpr float kLongStripCentralModuleT = 0.35f;
	static constexpr float kLongStripEndcapModuleT = 0.35f;

	// Module material properties - X0, L0, A, Z, Rho
	// Acts::Material pcMaterial(95.7, 465.2, 28.03, 14., 2.32e-3);
	static const Acts::Material kSiliconMaterial;

	static constexpr double kPstRadius = 200 * Acts::UnitConstants::mm;
	static constexpr double kPstHalfLengthZ = 2800 * Acts::UnitConstants::mm;

	static constexpr double kBeamPipeRadius = 19 * Acts::UnitConstants::mm;
	static constexpr double kBeamPipeHalfLengthZ = 70 * Acts::UnitConstants::cm;

	using ElementFactory = std::function<std::shared_ptr<ActsPlugins::TGeoDetectorElement>(
	    const ActsPlugins::TGeoDetectorElement::Identifier&, const TGeoNode&,
	    const TGeoMatrix& tGeoMatrix, ActsPlugins::TGeoAxes axes, double scalor,
	    std::shared_ptr<const Acts::ISurfaceMaterial> material)>;

	static std::shared_ptr<ActsPlugins::TGeoDetectorElement> defaultElementFactory(
	    const ActsPlugins::TGeoDetectorElement::Identifier& identifier, const TGeoNode& tGeoNode,
	    const TGeoMatrix& tGeoMatrix, ActsPlugins::TGeoAxes axes, double scalor,
	    std::shared_ptr<const Acts::ISurfaceMaterial> material);

	struct LayerConfig {
		std::string volumeName;
		std::vector<std::string> sensorNames;
		std::vector<std::pair<Acts::AxisDirection, std::pair<double, double>>> parseRanges;
		std::vector<std::pair<Acts::AxisDirection, double>> splitConfigs;
		std::vector<std::variant<int, Acts::BinningType>> binning0;
		std::vector<std::variant<int, Acts::BinningType>> binning1;
		std::pair<double, double> envelope
		    = { 1 * Acts::UnitConstants::mm, 1 * Acts::UnitConstants::mm };
		ActsPlugins::TGeoAxes localAxes = ActsPlugins::TGeoAxes("XYZ");
		std::shared_ptr<const Acts::ISurfaceMaterial> moduleMaterial = nullptr;
		bool isDSSD = false;
		double backsideGap = 0.0; // Abstand der Rückseite in ACTS-Einheiten (z.B. mm)
		double backsideStereo = 0.0;
	};

	struct Config {
		ElementFactory detectorElementFactory = defaultElementFactory;
		Acts::Logging::Level layerLogLevel = Acts::Logging::INFO;
		bool protoMaterial = false;
		std::size_t buildLevel = 3;
		std::string fileName;
		std::shared_ptr<const ActsPlugins::ITGeoIdentifierProvider> identifierProvider
		    = std::make_shared<MyIdentifierProvider>();
		std::vector<LayerConfig> layerConfigs;
		std::vector<LayerConfig> pxd_layerConfigs;
		std::shared_ptr<const Acts::IMaterialDecorator> materialDecorator = nullptr;
	};

	explicit Belle2Builder(const Config& cfg,
	    std::unique_ptr<const Acts::Logger> logger
	    = Acts::getDefaultLogger("GenDetBldr", Acts::Logging::INFO));

  protected:
	ProtoLayerCreator createPixelProtoLayerCreator();
	ProtoLayerCreator createShortStripProtoLayerCreator();
	ProtoLayerCreator createLongStripProtoLayerCreator();

	Config m_cfg;

	std::unique_ptr<const Acts::Logger> m_logger;
	const Acts::Logger& logger() const { return *m_logger; }

	std::shared_ptr<const Acts::ISurfaceMaterial> m_beamPipeMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_pstMaterial;

	std::shared_ptr<const Acts::ISurfaceMaterial> m_pixelCentralMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_pixelEndcapMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_pixelCentralModuleMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_pixelEndcapModuleMaterial;

	std::shared_ptr<const Acts::ISurfaceMaterial> m_shortStripCentralMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_shortStripEndcapMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_shortStripCentralModuleMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_shortStripEndcapModuleMaterial;

	std::shared_ptr<const Acts::ISurfaceMaterial> m_longStripCentralMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_longStripEndcapMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_longStripCentralModuleMaterial;
	std::shared_ptr<const Acts::ISurfaceMaterial> m_longStripEndcapModuleMaterial;
};

} // namespace ActsExamples::Generic

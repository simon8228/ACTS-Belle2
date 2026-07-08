// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/Belle2/Belle2.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/BlueprintNode.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp"
#include "Acts/Geometry/CylinderVolumeBounds.hpp"
#include "Acts/Geometry/CylinderVolumeBuilder.hpp"
#include "Acts/Geometry/CylinderVolumeHelper.hpp"
#include "Acts/Geometry/Extent.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/LayerArrayCreator.hpp"
#include "Acts/Geometry/LayerBlueprintNode.hpp"
#include "Acts/Geometry/LayerCreator.hpp"
#include "Acts/Geometry/MaterialDesignatorBlueprintNode.hpp"
#include "Acts/Geometry/NavigationPolicyFactory.hpp"
#include "Acts/Geometry/PassiveLayerBuilder.hpp"
#include "Acts/Geometry/ProtoLayer.hpp"
#include "Acts/Geometry/ProtoLayerHelper.hpp"
#include "Acts/Geometry/SurfaceArrayCreator.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/TrackingVolumeArrayCreator.hpp"
#include "Acts/Geometry/VolumeAttachmentStrategy.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/ProtoSurfaceMaterial.hpp"
#include "Acts/Navigation/CylinderNavigationPolicy.hpp"
#include "Acts/Navigation/MultiNavigationPolicy.hpp"
#include "Acts/Navigation/MyNavigationPolicy.hpp"
#include "Acts/Navigation/SurfaceArrayNavigationPolicy.hpp"
#include "Acts/Navigation/TryAllNavigationPolicy.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/RadialBounds.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Surfaces/ConvexPolygonBounds.hpp"
#include "Acts/Surfaces/TrapezoidBounds.hpp"
#include "Acts/Utilities/AxisDefinitions.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/Belle2/Belle2Element.hpp"
#include "ActsExamples/Belle2/CDCWireElement.hpp"
#include "ActsExamples/Belle2/LayerBuilder.hpp"
#include "ActsExamples/Belle2/MyIdentifierHook.hpp"
#include "ActsExamples/Belle2/MyIdentifierProvider.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include "ActsPlugins/Root/TGeoParser.hpp"

#include <cstddef>
#include <fstream>
#include <memory>

#include "./Belle2Builder.hpp"
#include "TGeoManager.h"

namespace ActsExamples {

namespace B2 {
namespace {

// Bin-count multiplier for all non-sensitive (support/material) layers.
// Set to 1 for production, 2/4/... to increase material map resolution.
constexpr int ns_factor = 1;

// Bin-count multiplier for PXD and SVD sensitive layers.
constexpr int s_factor = 1;

class Gen3Belle2Builder : public Belle2Builder {
 public:
  using Belle2Builder::Belle2Builder;

  std::unique_ptr<const Acts::TrackingGeometry> buildTrackingGeometry(
      const Acts::GeometryContext& gctx,
      std::optional<std::filesystem::path> graphvizFile);

  void buildSVD(Acts::Experimental::BlueprintNode& parent,
                const Acts::GeometryContext& gctx);

  void buildCDC(Acts::Experimental::BlueprintNode& parent,
                const Acts::GeometryContext& gctx);

  void buildPXD(Acts::Experimental::BlueprintNode& parent,
                     const Acts::GeometryContext& gctx);

  std::vector<ProtoLayerSurfaces> buildLayers(
      const Acts::GeometryContext& gctx);

  std::vector<ProtoLayerSurfaces> buildCDCLayers(
      const Acts::GeometryContext& gctx);

  std::vector<ProtoLayerSurfaces> buildPXDLayers(
      const Acts::GeometryContext& gctx);

  static std::shared_ptr<const Acts::HomogeneousSurfaceMaterial> asHomogeneous(
      std::string_view debugLabel,
      const std::shared_ptr<const Acts::ISurfaceMaterial>& material) {
    auto hm = std::dynamic_pointer_cast<const Acts::HomogeneousSurfaceMaterial>(
        material);
    if (hm == nullptr) {
      throw std::runtime_error(std::string{debugLabel} +
                               " material is not homogeneous");
    }
    return hm;
  }

  using LayerType = Acts::SurfaceArrayNavigationPolicy::LayerType;
  std::unique_ptr<Acts::NavigationPolicyFactory> createNavigationPolicy(
      LayerType layerType,
      std::pair<std::size_t, std::size_t> bins = {0, 0}) const {
    using SrfArrayNavPol = Acts::SurfaceArrayNavigationPolicy;
    return Acts::NavigationPolicyFactory{}
        .add<Acts::CylinderNavigationPolicy>()
        .add<SrfArrayNavPol>(
            SrfArrayNavPol::Config{.layerType = layerType, .bins = bins})
        .asUniquePtr();
  }
  std::unique_ptr<Acts::NavigationPolicyFactory> createNavigationPolicy2()
      const {
    return Acts::NavigationPolicyFactory{}
        .add<Acts::TryAllNavigationPolicy>()
        .asUniquePtr();
  }
  std::unique_ptr<Acts::NavigationPolicyFactory> createNavigationPolicy3(
      LayerType layerType,
      std::pair<std::size_t, std::size_t> bins = {0, 0}) const {
    using SrfArrayNavPol = Acts::SurfaceArrayNavigationPolicy;
    return Acts::NavigationPolicyFactory{}
        .add<SrfArrayNavPol>(
            SrfArrayNavPol::Config{.layerType = layerType, .bins = bins})
        .add<Acts::TryAllNavigationPolicy>()
        .asUniquePtr();
  }
  std::unique_ptr<Acts::NavigationPolicyFactory> createNavigationPolicy4(
      LayerType layerType,
      std::pair<std::size_t, std::size_t> bins = {0, 0}) const {
    return Acts::NavigationPolicyFactory{}
        .add<Acts::MyNavigationPolicy>()
        .asUniquePtr();
  }
  const std::vector<std::shared_ptr<ActsPlugins::TGeoDetectorElement>>&
  getElementStore() const {
    return m_elementStore;
  }

  const std::vector<std::shared_ptr<CDCWireElement>>& getCDCElementStore()
      const {
    return m_cdcElementStore;
  }

 private:
  std::vector<std::shared_ptr<ActsPlugins::TGeoDetectorElement>> m_elementStore;

  std::vector<std::shared_ptr<CDCWireElement>> m_cdcElementStore;
};

std::unique_ptr<const Acts::TrackingGeometry>
Gen3Belle2Builder::buildTrackingGeometry(
    const Acts::GeometryContext& gctx,
    std::optional<std::filesystem::path> graphvizFile) {
  using enum Acts::AxisDirection;
  using namespace Acts::UnitLiterals;
  using namespace Acts::Experimental;
  using enum Acts::AxisBoundaryType;
  using enum Acts::CylinderVolumeBounds::Face;
  ACTS_INFO("Belle2 construction in  Gen3 mode");

  TGeoManager::Import("/home/configs/Belle2_Beampipe_PXD_SVD.root");
  std::string cdcFile = "wireInfo.csv";

  Blueprint::Config cfg;
  cfg.envelope = Acts::ExtentEnvelope{{
      .z = {20_mm, 20_mm},
      .r = {0_mm, 20_mm},
  }};
  cfg.hook = std::make_shared<MyIdentifierHook>();
  Blueprint root{cfg};

  root.addCylinderContainer("Detector", AxisR, [&](auto& detector) {
    // auto beampipeBounds = std::make_unique<Acts::CylinderVolumeBounds>(0_mm,
    // kBeamPipeRadius, kBeamPipeHalfLengthZ); auto beampipe =
    // std::make_unique<Acts::TrackingVolume>(Acts::Transform3::Identity(),
    // std::move(beampipeBounds), "Beampipe");
    /*
    detector.addMaterial("BeampipeMaterial", [&](auto& bpMat) {
        bpMat.configureFace(OuterCylinder, asHomogeneous("Beam pipe",
    m_beamPipeMaterial)); bpMat.addStaticVolume(std::move(beampipe));
    });
    */
    buildBeamPipe(detector, gctx);
    buildPXD(detector, gctx);
    buildSVD(detector, gctx);
    // buildCDC(detector, gctx);
    // buildCDC(detector, gctx);
  });

  if (graphvizFile) {
    std::ofstream file(*graphvizFile);
    root.graphviz(file);
  }

  BlueprintOptions options;
  auto trackingGeometry =
      root.construct(options, gctx, *logger().clone("Blprnt"));
  ACTS_INFO("Lade material_map.json direkt in C++...");

  return trackingGeometry;
}

void Gen3Belle2Builder::buildCDC(Acts::Experimental::BlueprintNode& parent,
                                 const Acts::GeometryContext& gctx) {
  using namespace Acts::Experimental;
  using namespace Acts::UnitLiterals;
  using enum Acts::CylinderVolumeBounds::Face;

  ACTS_DEBUG("Building CDC");
  auto cdcLayers = buildCDCLayers(gctx);

  auto cdcBounds =
      std::make_shared<Acts::CylinderVolumeBounds>(160_mm, 3000_mm, 3000_mm);

  struct TrackingVolumeAccessor
      : public Acts::Experimental::StaticBlueprintNode {
    static Acts::TrackingVolume* getVolume(
        Acts::Experimental::StaticBlueprintNode& node) {
      return static_cast<TrackingVolumeAccessor&>(node).m_volume.get();
    }
  };
  parent.addMaterial("CDC_PassiveMaterial", [&](auto& matNode) {
    using Face = Acts::CylinderVolumeBounds::Face;
    using enum Acts::AxisDirection;
    using enum Acts::AxisBoundaryType;

    matNode.addStaticVolume(
        Acts::Transform3::Identity(), cdcBounds, "CDC_Container",
        [&](auto& cdcVol) {
          for (size_t i = 0; i < cdcLayers.size(); ++i) {
            std::string name = "CDC_Layer" + std::to_string(i);

            cdcVol.addLayer(name, [&](auto& layer) {
              layer.setProtoLayer(cdcLayers[i].protoLayer);
              layer.setEnvelope(
                  Acts::ExtentEnvelope{{.z = {5_mm, 5_mm}, .r = {0_mm, 0_mm}}});
              layer.setNavigationPolicyFactory(createNavigationPolicy2());
            });
          }

          cdcVol.addLayer("CDC_SuppDisk1", [&](auto& layer) {
            double discZ = -700.0_mm;
            auto bounds =
                std::make_shared<Acts::RadialBounds>(200.0_mm, 1200.0_mm);
            auto transform = Acts::Transform3(Acts::Translation3(0, 0, discZ));
            auto surf =
                Acts::Surface::makeShared<Acts::DiscSurface>(transform, bounds);

            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 150, 200.0_mm, 1200.0_mm, Acts::open,
                                   Acts::AxisDirection::AxisR);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));

            Acts::MutableProtoLayer pl(gctx, {surf});
            layer.setProtoLayer(pl);
            layer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {0_mm, 0_mm}}});
          });
          cdcVol.addLayer("CDC_SuppDisk2", [&](auto& layer) {
            double discZ = 1500.0_mm;
            auto bounds =
                std::make_shared<Acts::RadialBounds>(300.0_mm, 1200.0_mm);
            auto transform = Acts::Transform3(Acts::Translation3(0, 0, discZ));
            auto surf =
                Acts::Surface::makeShared<Acts::DiscSurface>(transform, bounds);

            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 200, 400.0_mm, 1200.0_mm, Acts::open,
                                   Acts::AxisDirection::AxisR);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));

            Acts::MutableProtoLayer pl(gctx, {surf});
            layer.setProtoLayer(pl);
            layer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {0_mm, 0_mm}}});
          });

          cdcVol.setNavigationPolicyFactory(createNavigationPolicy2());
        });
  });
}

std::vector<ProtoLayerSurfaces> Gen3Belle2Builder::buildCDCLayers(
    const Acts::GeometryContext& gctx) {
  std::vector<ProtoLayerSurfaces> cpLayers;

  using LayerSurfaceVector = std::vector<std::shared_ptr<const Acts::Surface>>;
  LayerSurfaceVector layerSurfaces;
  const auto& layerConfigs = m_cfg.layerConfigs;

  ACTS_DEBUG(" layers : found " << layerConfigs.size() << " configuration(s)");

  // Helper function to fill the layer
  auto fillLayer =
      [&](const std::vector<std::shared_ptr<Acts::Surface>>& sVector) {
        std::size_t nb0 = 1;
        std::size_t nb1 = 1;

        Acts::MutableProtoLayer pl(gctx, sVector);

        ACTS_DEBUG("- creating CylinderLayer with "
                   << sVector.size() << " surfaces at r = "
                   << pl.medium(Acts::AxisDirection::AxisR));

        pl.envelope[Acts::AxisDirection::AxisR] = {0, 0};
        pl.envelope[Acts::AxisDirection::AxisZ] = {5, 5};
        ProtoLayerSurfaces pls{std::move(pl), sVector, nb0, nb1};
        cpLayers.push_back(std::move(pls));
      };

  std::map<unsigned int, std::vector<std::shared_ptr<Acts::Surface>>>
      surfacesByLayer;

  std::ifstream file("/home/configs/wireInfo.csv");
  if (!file.is_open()) {
    ACTS_ERROR("Konnte wireInfo.csv nicht oeffnen!");
    return cpLayers;
  }

  std::string line;
  bool isHeader = true;

  while (std::getline(file, line)) {
    if (isHeader) {
      isHeader = false;
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> tokens;

    while (std::getline(ss, cell, ',')) {
      tokens.push_back(cell);
    }

    if (tokens.size() < 18)
      continue;

    unsigned int layerIndex = std::stoul(tokens[1]);
    unsigned int wireIndex = std::stoul(tokens[2]);

    double halfLength = std::stod(tokens[4]);
    double cx = std::stod(tokens[5]);
    double cy = std::stod(tokens[6]);
    double cz = std::stod(tokens[7]);

    double r00 = std::stod(tokens[8]), r10 = std::stod(tokens[9]),
           r20 = std::stod(tokens[10]);
    double r01 = std::stod(tokens[11]), r11 = std::stod(tokens[12]),
           r21 = std::stod(tokens[13]);
    double r02 = std::stod(tokens[14]), r12 = std::stod(tokens[15]),
           r22 = std::stod(tokens[16]);
    double cell_size = std::abs(std::stod(tokens[17]));
    if (cell_size < 1.0)
      cell_size = 18.0;

    Acts::GeometryIdentifier geoId = Acts::GeometryIdentifier()
                                         .withVolume(14)
                                         .withLayer(layerIndex + 1)
                                         .withSensitive(wireIndex + 1);

    auto bounds = std::make_shared<const Acts::LineBounds>(
        (cell_size / 2.0) - 1.2,
        halfLength);  // TODO: Check for overlapping, this causes holes in the
                      // trajectory
    // Right now maximum of ~3 holes in 56 layers, but check it

    ACTS_FATAL(layerIndex << " " << cell_size);
    Acts::Transform3 transform = Acts::Transform3::Identity();

    Acts::RotationMatrix3 rot;
    rot << r00, r01, r02, r10, r11, r12, r20, r21, r22;
    transform.linear() = rot;
    transform.translation() = Acts::Vector3(cx, cy, cz);

    auto wireElement =
        std::make_shared<CDCWireElement>(geoId, transform, bounds, 1.0);
    wireElement->surface().assignGeometryId(geoId);

    surfacesByLayer[layerIndex].push_back(
        wireElement->surface().getSharedPtr());
    m_cdcElementStore.push_back(wireElement);
  }

  for (auto const& [layerIdx, surfaces] : surfacesByLayer) {
    ACTS_DEBUG("- Custom CSV Drift Chamber loaded. Found "
              << surfaces.size() << " wires for CDC layer " << layerIdx);

    Acts::MutableProtoLayer probe(gctx, surfaces);
    double layerR  = probe.medium(Acts::AxisDirection::AxisR);
    double layerCz = probe.medium(Acts::AxisDirection::AxisZ);
    double halfZ   = 0.5 * probe.range(Acts::AxisDirection::AxisZ);

    auto cylBounds = std::make_shared<Acts::CylinderBounds>(layerR, halfZ);
    Acts::Transform3 cylTransform = Acts::Transform3::Identity();
    cylTransform.translation() = Acts::Vector3(0, 0, layerCz);
    auto passiveSurf = Acts::Surface::makeShared<Acts::CylinderSurface>(
        cylTransform, cylBounds);

    Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                        Acts::AxisDirection::AxisPhi);
    bu += Acts::BinUtility(ns_factor * 350, -halfZ, halfZ, Acts::open,
                           Acts::AxisDirection::AxisZ);
    passiveSurf->assignSurfaceMaterial(
        std::make_shared<Acts::ProtoSurfaceMaterial>(bu));

    auto layerSurfaces = surfaces;
    layerSurfaces.push_back(passiveSurf);
    fillLayer(layerSurfaces);
  }

  return cpLayers;
}

void Gen3Belle2Builder::buildBeamPipe(Acts::Experimental::BlueprintNode& parent,
                                      const Acts::GeometryContext& gctx) {
  using namespace Acts::Experimental;
  using namespace Acts::UnitLiterals;
  using enum Acts::CylinderVolumeBounds::Face;

  ACTS_DEBUG("Building Beam Pipe");

  double beamPipeRadius = 13.0_mm;
  double beamPipeHalfZ = 110.0_mm;
  // Inner material surface placed just inside the Be wall.
  // halfZ=60mm is large enough for all tracks in eta=[-1.5,+1.9]
  // (z_intersection = r/tan(theta) <= 10.5/tan(17deg) ~ 34mm << 60mm).
  // Using a dedicated surface avoids adjustBinUtility expanding to +-3000mm,
  // which previously caused the single HomogeneousSurface to average over all
  // eta tracks and over-estimate material at central eta by ~8x.
  double bpMatRadius = 10.5_mm;
  double bpMatHalfZ = 90.0_mm;

  auto bpBounds = std::make_shared<Acts::CylinderVolumeBounds>(
      0_mm, beamPipeRadius, beamPipeHalfZ);

  parent.addMaterial("BeamPipe_Material", [&](auto& matNode) {
    matNode.addStaticVolume(
        Acts::Transform3::Identity(), bpBounds, "BeamPipe_Volume",
        [&](auto& bpVol) {
          bpVol.addLayer("BeamPipe_Layer", [&](auto& bpLayer) {
            auto bpSurfBounds =
                std::make_shared<Acts::CylinderBounds>(bpMatRadius, bpMatHalfZ);
            auto bpSurf = Acts::Surface::makeShared<Acts::CylinderSurface>(
                Acts::Transform3::Identity(), bpSurfBounds);

            if (m_cfg.protoMaterial) {
              Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                  Acts::AxisDirection::AxisPhi);
              bu += Acts::BinUtility(ns_factor * 100, -bpMatHalfZ, bpMatHalfZ, Acts::open,
                                     Acts::AxisDirection::AxisZ);
              bpSurf->assignSurfaceMaterial(
                  std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            } else {
              bpSurf->assignSurfaceMaterial(m_beamPipeMaterial);
            }

            std::vector<std::shared_ptr<Acts::Surface>> bpSurfaces = {bpSurf};
            Acts::MutableProtoLayer pl(gctx, bpSurfaces);
            bpLayer.setProtoLayer(pl);
            bpLayer.setEnvelope(Acts::ExtentEnvelope{
                {.z = {5_mm, 5_mm}, .r = {0.5_mm, 0.5_mm}}});
            bpLayer.setNavigationPolicyFactory(createNavigationPolicy2());
          });
          bpVol.setNavigationPolicyFactory(createNavigationPolicy2());
        });
  });
}

void Gen3Belle2Builder::buildPXD(Acts::Experimental::BlueprintNode& parent,
                                      const Acts::GeometryContext& gctx) {
  using namespace Acts::Experimental;
  using namespace Acts::UnitLiterals;
  using enum Acts::CylinderVolumeBounds::Face;

  ACTS_DEBUG("Building PXD");

  auto pxdLayers = buildPXDLayers(gctx);

  double pxdRadius = 30.0_mm;
  double pxdHalfZ = 90.0_mm;
  Acts::Transform3 pxdTransform = Acts::Transform3::Identity();
  pxdTransform.translation() = Acts::Vector3(0, 0, 15.0_mm); // Find the correct z-offset of the PXD

  auto pxdBounds = std::make_shared<Acts::CylinderVolumeBounds>(
      13.0_mm, pxdRadius, 120.0_mm);


  parent.addMaterial("PXD_PassiveMaterial", [&](auto& matNode) {
    using Face = Acts::CylinderVolumeBounds::Face;
    using enum Acts::AxisDirection;
    using enum Acts::AxisBoundaryType;

    // matNode.configureFace(
    //     Face::OuterCylinder,
    //     Acts::DirectedProtoAxis(AxisRPhi, Closed, -M_PI, M_PI, 120),
    //     Acts::DirectedProtoAxis(AxisZ, Open, -100_mm, 120_mm, 110));

    matNode.addStaticVolume(
        pxdTransform, pxdBounds, "PXD_Container",
        [&](auto& pxdVol) {
          pxdVol.addLayer("PXD_suppLayer1", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 20_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(18_mm, 80_mm);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 320, -80_mm, 80_mm, Acts::open, Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          // pxdVol.addLayer("PXD_suppLayer4", [&](auto& suppLayer) {
          //   Acts::Transform3 t = Acts::Transform3::Identity();
          //   t.translation() = Acts::Vector3(0, 0, 20_mm);
          //   auto bounds = std::make_shared<Acts::CylinderBounds>(16_mm, 80_mm);
          //   auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
          //   Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
          //   bu += Acts::BinUtility(320, -80_mm, 80_mm, Acts::open, Acts::AxisDirection::AxisZ);
          //   surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
          //   Acts::MutableProtoLayer pl(gctx, {surf});
          //   suppLayer.setProtoLayer(pl);
          //   suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          // });
          pxdVol.addLayer("PXD_suppLayer4", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, -35_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(16_mm, 12.5_mm);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 30, -12.5_mm, 12.5_mm, Acts::open, Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          pxdVol.addLayer("PXD_suppLayer5", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 65_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(16_mm, 12.5_mm);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 30, -12.5_mm, 12.5_mm, Acts::open, Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          pxdVol.addLayer("PXD_suppLayer2", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 20_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(12.5_mm, 80_mm);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 80, -80_mm, 80_mm, Acts::open, Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          pxdVol.addLayer("PXD_suppLayer3", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 20_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(30_mm, 80_mm);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed, Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 80, -80_mm, 80_mm, Acts::open, Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });

          for (int i = 0; i < 2; ++i) {
            std::string name = "PXD_Layer" + std::to_string(i + 1);

            pxdVol.addLayer(name, [&](auto& layer) {
              layer.setProtoLayer(pxdLayers[i].protoLayer);

              layer.setEnvelope(
                  Acts::ExtentEnvelope{{.z = {5_mm, 5_mm}, .r = {5_mm, 5_mm}}});

              layer.setNavigationPolicyFactory(createNavigationPolicy3(
                  LayerType::Cylinder,
                  std::make_pair(pxdLayers[i].bins0, pxdLayers[i].bins1)));
            });
          }

          pxdVol.setNavigationPolicyFactory(createNavigationPolicy2());
        });
  });
}

std::vector<ProtoLayerSurfaces> Gen3Belle2Builder::buildPXDLayers(
    const Acts::GeometryContext& gctx) {
  std::vector<ProtoLayerSurfaces> cpLayers;

  // Bail out if you have no gGeoManager
  if (gGeoManager == nullptr) {
    ACTS_WARNING("No gGeoManager found - bailing out.");
    return {};
  }

  using LayerSurfaceVector = std::vector<std::shared_ptr<const Acts::Surface>>;
  LayerSurfaceVector layerSurfaces;
  const auto& layerConfigs = m_cfg.pxd_layerConfigs;
  // Screen output of the configuration
  ACTS_DEBUG(" layers : found " << layerConfigs.size() << " configuration(s)");

  // Helper function to fill the layer
  auto fillLayer =
      [&](const std::vector<std::shared_ptr<Acts::Surface>>& sVector,
          const LayerConfig& lCfg) {
        std::size_t nb0 = std::get<int>(lCfg.binning0.at(0));
        std::size_t nb1 = std::get<int>(lCfg.binning1.at(0));
        ;

        Acts::MutableProtoLayer pl(gctx, sVector);

        ACTS_DEBUG("- creating CylinderLayer with "
                   << sVector.size() << " surfaces at r = "
                   << pl.medium(Acts::AxisDirection::AxisR));

        pl.envelope[Acts::AxisDirection::AxisR] = {lCfg.envelope.first,
                                                   lCfg.envelope.second};
        pl.envelope[Acts::AxisDirection::AxisZ] = {lCfg.envelope.second,
                                                   lCfg.envelope.second};
        ProtoLayerSurfaces pls{std::move(pl), sVector, nb0, nb1};
        cpLayers.push_back(std::move(pls));
      };

  // Loop over Layer configs to loop over layers
  for (const auto& layerCfg : layerConfigs) {
    ACTS_DEBUG("- layer configuration found for layer " << layerCfg.volumeName
                                                        << " with sensors ");
    std::vector<std::shared_ptr<Acts::Surface>> sVector;

    for (auto& sensor : layerCfg.sensorNames) {
      ACTS_DEBUG("  - sensor: " << sensor);
    }
    if (!layerCfg.parseRanges.empty()) {
      for (const auto& [axisDir, pRange] : layerCfg.parseRanges) {
        ACTS_DEBUG("- layer parsing restricted in "
                   << axisDirectionName(axisDir) << " to [" << pRange.first
                   << "/" << pRange.second << "].");
      }
    }
    if (!layerCfg.splitConfigs.empty()) {
      for (const auto& [axisDir, tConfig] : layerCfg.splitConfigs) {
        ACTS_DEBUG("- layer splitting attempt in " << axisDirectionName(axisDir)
                                                   << " with tolerance "
                                                   << tConfig << ".");
      }
    }

    // Either pick the configured volume or take the top level volume
    // and retrieve its global transformation.
    TGeoVolume* tVolume =
        gGeoManager->FindVolumeFast(layerCfg.volumeName.c_str());
    TGeoHMatrix gmatrix = TGeoIdentity((layerCfg.volumeName + "ID").c_str());

    if (tVolume == nullptr) {
      tVolume = gGeoManager->GetTopVolume();
      ACTS_DEBUG("- search volume is TGeo top volume");
    } else {
      ACTS_DEBUG("- setting search volume to " << tVolume->GetName());

      auto node = ActsPlugins::TGeoParser::findNodeRecursive(
          gGeoManager->GetTopNode(), tVolume->GetName());

      if (node == nullptr) {
        std::string volname(tVolume->GetName());
        throw std::invalid_argument("Could not locate node for " + volname);
      }

      gmatrix = *(node->GetMatrix());
    }

    if (tVolume != nullptr) {
      ActsPlugins::TGeoParser::Options tgpOptions;
      tgpOptions.volumeNames = {layerCfg.volumeName};
      tgpOptions.targetNames = layerCfg.sensorNames;
      // tgpOptions.parseRanges = layerCfg.parseRanges;
      tgpOptions.unit = 1 * Acts::UnitConstants::cm;
      ActsPlugins::TGeoParser::State tgpState;
      tgpState.volume = tVolume;

      ACTS_DEBUG("- applying  " << layerCfg.parseRanges.size()
                                << " search restrictions.");
      for (const auto& [axisDir, pRange] : layerCfg.parseRanges) {
        ACTS_VERBOSE(" - range " << axisDirectionName(axisDir) << " within [ "
                                 << pRange.first << ", " << pRange.second
                                 << "]");
      }

      ActsPlugins::TGeoParser::select(tgpState, tgpOptions, gmatrix);

      ACTS_DEBUG("- number of selected nodes found : "
                 << tgpState.selectedNodes.size());

      for (auto& snode : tgpState.selectedNodes) {
        auto identifier =
            m_cfg.identifierProvider != nullptr
                ? m_cfg.identifierProvider->identify(gctx, *snode.node)
                : ActsPlugins::TGeoDetectorElement::Identifier();

        auto tgElementFront = m_cfg.detectorElementFactory(
            identifier, *snode.node, *snode.transform, layerCfg.localAxes,
            1 * Acts::UnitConstants::cm, layerCfg.moduleMaterial);

        m_elementStore.push_back(tgElementFront);

        auto sensorSurface = tgElementFront->surface().getSharedPtr();
        ACTS_DEBUG("PXD sensor center: z=" << sensorSurface->center(gctx).z()
                   << " bounds type=" << sensorSurface->bounds().type());

        if (const auto* rb = dynamic_cast<const Acts::RectangleBounds*>(
                &sensorSurface->bounds())) {
          Acts::BinUtility bu(s_factor * 15, -rb->halfLengthX(), rb->halfLengthX(),
                              Acts::open, Acts::AxisDirection::AxisX);
          bu += Acts::BinUtility(s_factor * 20, -rb->halfLengthY(), rb->halfLengthY(),
                                 Acts::open, Acts::AxisDirection::AxisY);
          sensorSurface->assignSurfaceMaterial(
              std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
        }

        sVector.push_back(tgElementFront->surface().getSharedPtr());
      }

      ACTS_DEBUG("- created TGeoDetectorElements : " << sVector.size());

      fillLayer(sVector, layerCfg);
    }
  }
  return cpLayers;
}

void Gen3Belle2Builder::buildSVD(Acts::Experimental::BlueprintNode& parent,
                                 const Acts::GeometryContext& gctx) {
  using namespace Acts::Experimental;
  using namespace Acts::UnitLiterals;
  using enum Acts::CylinderVolumeBounds::Face;

  ACTS_DEBUG("Building SVD with Nested Volumes");

  // ProtoLayerCreator svdCreator = createPixelProtoLayerCreator();
  // Hier muss das root plugin angebunden werden
  auto svdLayers = buildLayers(gctx);

  double z_positionL6 = 55_mm;
  double z_positionL5 = 47.5_mm;
  double z_positionL4 = 24.5_mm;
  double z_positionL3 = 29.5_mm;

  std::vector<double> z_ranges = {140.0_mm, 300.0_mm, 400.0_mm, 500.0_mm};

  auto svdBounds =
      std::make_shared<Acts::CylinderVolumeBounds>(30_mm, 150_mm, 3000_mm);

  Acts::Transform3 transformationZL6 = Acts::Transform3::Identity();
  transformationZL6.translation() = Acts::Vector3(0_mm, 0_mm, z_positionL6);
  Acts::Transform3 transformationZL5 = Acts::Transform3::Identity();
  transformationZL5.translation() = Acts::Vector3(0_mm, 0_mm, z_positionL5);
  Acts::Transform3 transformationZL4 = Acts::Transform3::Identity();
  transformationZL4.translation() = Acts::Vector3(0_mm, 0_mm, z_positionL4);
  Acts::Transform3 transformationZL3 = Acts::Transform3::Identity();
  transformationZL3.translation() = Acts::Vector3(0_mm, 0_mm, z_positionL3);

  std::vector<Acts::Transform3> z_transformations = {
      transformationZL3, transformationZL4, transformationZL5,
      transformationZL6};

  parent.addMaterial("SVD_PassiveMaterial", [&](auto& matNode) {
    using Face = Acts::CylinderVolumeBounds::Face;
    using enum Acts::AxisDirection;
    using enum Acts::AxisBoundaryType;

    matNode.addStaticVolume(
        Acts::Transform3::Identity(), svdBounds, "SVD_Container",
        [&](auto& svdVol) {
          svdVol.addLayer("SVD_suppLayer1", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, -50_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(50_mm, 100);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 250, -250, 250, Acts::open,
                                   Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          svdVol.addLayer("SVD_suppLayer2", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 30_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(110_mm, 150);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 75, -150, 150, Acts::open,
                                   Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });
          svdVol.addLayer("SVD_suppLayer3", [&](auto& suppLayer) {
            Acts::Transform3 t = Acts::Transform3::Identity();
            t.translation() = Acts::Vector3(0, 0, 100_mm);
            auto bounds = std::make_shared<Acts::CylinderBounds>(155_mm, 500);
            auto surf = Acts::Surface::makeShared<Acts::CylinderSurface>(t, bounds);
            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 250, -500, 500, Acts::open,
                                   Acts::AxisDirection::AxisZ);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            Acts::MutableProtoLayer pl(gctx, {surf});
            suppLayer.setProtoLayer(pl);
            suppLayer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {1_mm, 1_mm}}});
          });

          svdVol.addLayer("SVD_SuppDisk1", [&](auto& layer) {
            double discZ = -300.0_mm;
            auto bounds =
                std::make_shared<Acts::RadialBounds>(60.0_mm, 130.0_mm);
            auto transform = Acts::Transform3(Acts::Translation3(0, 0, discZ));
            auto surf =
                Acts::Surface::makeShared<Acts::DiscSurface>(transform, bounds);

            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 20, 60.0_mm, 130.0_mm, Acts::open,
                                   Acts::AxisDirection::AxisR);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));

            Acts::MutableProtoLayer pl(gctx, {surf});
            layer.setProtoLayer(pl);
            layer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {0_mm, 0_mm}}});
          });
          svdVol.addLayer("SVD_SuppDisk2", [&](auto& layer) {
            double discZ = 380.0_mm;
            auto bounds =
                std::make_shared<Acts::RadialBounds>(35.0_mm, 135.0_mm);
            auto transform = Acts::Transform3(Acts::Translation3(0, 0, discZ));
            auto surf =
                Acts::Surface::makeShared<Acts::DiscSurface>(transform, bounds);

            Acts::BinUtility bu(72, -M_PI, M_PI, Acts::closed,
                                Acts::AxisDirection::AxisPhi);
            bu += Acts::BinUtility(ns_factor * 25, 35.0_mm, 135.0_mm, Acts::open,
                                   Acts::AxisDirection::AxisR);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));

            Acts::MutableProtoLayer pl(gctx, {surf});
            layer.setProtoLayer(pl);
            layer.setEnvelope(
                Acts::ExtentEnvelope{{.z = {1_mm, 1_mm}, .r = {0_mm, 0_mm}}});
          });

          for (int i = 0; i < 4; ++i) {
            std::string name = "SVD_Layer" + std::to_string(i + 3);

            svdVol.addLayer(name, [&](auto& layer) {
              layer.setProtoLayer(svdLayers[i].protoLayer);

              layer.setEnvelope(
                  Acts::ExtentEnvelope{{.z = {5_mm, 5_mm}, .r = {5_mm, 5_mm}}});
              /*
                            layer.setNavigationPolicyFactory(createNavigationPolicy3(
                                LayerType::Cylinder,
                                {svdLayers[i].bins0, svdLayers[i].bins1}));
                          */
              layer.setNavigationPolicyFactory(createNavigationPolicy2());
            });
          }

          for (int i = 4; i < 7; ++i) {
            std::string name = "SVD_SlantedLayer" + std::to_string(i);

            svdVol.addLayer(name, [&](auto& layer) {
              layer.setProtoLayer(svdLayers[i].protoLayer);

              layer.setEnvelope(
                  Acts::ExtentEnvelope{{.z = {5_mm, 5_mm}, .r = {5_mm, 5_mm}}});

              layer.setNavigationPolicyFactory(createNavigationPolicy3(
                  LayerType::Disc,
                  {svdLayers[i].bins0, svdLayers[i].bins1}));
            });
          }
          svdVol.setNavigationPolicyFactory(createNavigationPolicy2());
        });
  });
}

std::vector<ProtoLayerSurfaces> Gen3Belle2Builder::buildLayers(
    const Acts::GeometryContext& gctx) {
  std::vector<ProtoLayerSurfaces> cpLayers;

  // Bail out if you have no gGeoManager
  if (gGeoManager == nullptr) {
    ACTS_WARNING("No gGeoManager found - bailing out.");
    return {};
  }

  using LayerSurfaceVector = std::vector<std::shared_ptr<const Acts::Surface>>;
  LayerSurfaceVector layerSurfaces;
  const auto& layerConfigs = m_cfg.layerConfigs;
  // Screen output of the configuration
  ACTS_DEBUG(" layers : found " << layerConfigs.size() << " configuration(s)");

  // Helper function to fill the layer
  auto fillLayer =
      [&](const std::vector<std::shared_ptr<Acts::Surface>>& sVector,
          const LayerConfig& lCfg) {
        std::size_t nb0 = std::get<int>(lCfg.binning0.at(0));
        std::size_t nb1 = std::get<int>(lCfg.binning1.at(0));
        ;

        Acts::MutableProtoLayer pl(gctx, sVector);

        ACTS_DEBUG("- creating CylinderLayer with "
                   << sVector.size() << " surfaces at r = "
                   << pl.medium(Acts::AxisDirection::AxisR));

        pl.envelope[Acts::AxisDirection::AxisR] = {lCfg.envelope.first,
                                                   lCfg.envelope.second};
        pl.envelope[Acts::AxisDirection::AxisZ] = {lCfg.envelope.second,
                                                   lCfg.envelope.second};
        ProtoLayerSurfaces pls{std::move(pl), sVector, nb0, nb1};
        cpLayers.push_back(std::move(pls));
      };

  // Loop over Layer configs to loop over layers
  for (const auto& layerCfg : layerConfigs) {
    ACTS_DEBUG("- layer configuration found for layer " << layerCfg.volumeName
                                                        << " with sensors ");
    std::vector<std::shared_ptr<Acts::Surface>> sVector;

    for (auto& sensor : layerCfg.sensorNames) {
      ACTS_DEBUG("  - sensor: " << sensor);
    }
    if (!layerCfg.parseRanges.empty()) {
      for (const auto& [axisDir, pRange] : layerCfg.parseRanges) {
        ACTS_DEBUG("- layer parsing restricted in "
                   << axisDirectionName(axisDir) << " to [" << pRange.first
                   << "/" << pRange.second << "].");
      }
    }
    if (!layerCfg.splitConfigs.empty()) {
      for (const auto& [axisDir, tConfig] : layerCfg.splitConfigs) {
        ACTS_DEBUG("- layer splitting attempt in " << axisDirectionName(axisDir)
                                                   << " with tolerance "
                                                   << tConfig << ".");
      }
    }

    // Either pick the configured volume or take the top level volume
    // and retrieve its global transformation.
    TGeoVolume* tVolume =
        gGeoManager->FindVolumeFast(layerCfg.volumeName.c_str());
    TGeoHMatrix gmatrix = TGeoIdentity((layerCfg.volumeName + "ID").c_str());

    if (tVolume == nullptr) {
      tVolume = gGeoManager->GetTopVolume();
      ACTS_DEBUG("- search volume is TGeo top volume");
    } else {
      ACTS_DEBUG("- setting search volume to " << tVolume->GetName());

      auto node = ActsPlugins::TGeoParser::findNodeRecursive(
          gGeoManager->GetTopNode(), tVolume->GetName());

      if (node == nullptr) {
        std::string volname(tVolume->GetName());
        throw std::invalid_argument("Could not locate node for " + volname);
      }

      gmatrix = *(node->GetMatrix());
    }

    if (tVolume != nullptr) {
      ActsPlugins::TGeoParser::Options tgpOptions;
      tgpOptions.volumeNames = {layerCfg.volumeName};
      tgpOptions.targetNames = layerCfg.sensorNames;
      // tgpOptions.parseRanges = layerCfg.parseRanges;
      tgpOptions.unit = 1 * Acts::UnitConstants::cm;
      ActsPlugins::TGeoParser::State tgpState;
      tgpState.volume = tVolume;

      ACTS_DEBUG("- applying  " << layerCfg.parseRanges.size()
                                << " search restrictions.");
      for (const auto& [axisDir, pRange] : layerCfg.parseRanges) {
        ACTS_VERBOSE(" - range " << axisDirectionName(axisDir) << " within [ "
                                 << pRange.first << ", " << pRange.second
                                 << "]");
      }

      ActsPlugins::TGeoParser::select(tgpState, tgpOptions, gmatrix);

      ACTS_DEBUG("- number of selected nodes found : "
                 << tgpState.selectedNodes.size());

      for (auto& snode : tgpState.selectedNodes) {
        auto identifier =
            m_cfg.identifierProvider != nullptr
                ? m_cfg.identifierProvider->identify(gctx, *snode.node)
                : ActsPlugins::TGeoDetectorElement::Identifier();

        auto tgElementFront = m_cfg.detectorElementFactory(
            identifier, *snode.node, *snode.transform, layerCfg.localAxes,
            1 * Acts::UnitConstants::cm, layerCfg.moduleMaterial);

        m_elementStore.push_back(tgElementFront);

        auto sensorSurface = tgElementFront->surface().getSharedPtr();

        ACTS_DEBUG("- sensor " << layerCfg.volumeName << " bounds type : "
                               << sensorSurface->bounds().type() << " -> "
                               << sensorSurface->bounds());

        if (sensorSurface->bounds().type() ==
            Acts::SurfaceBounds::eConvexPolygon) {
          Acts::MaterialSlab svdSlab(B2::Belle2Builder::kSiliconMaterial,
                                     0.32 * Acts::UnitConstants::mm);
          //   sensorSurface->assignSurfaceMaterial(
          //       std::make_shared<Acts::HomogeneousSurfaceMaterial>(svdSlab));
          sensorSurface->assignSurfaceMaterial(nullptr);

        } else {
          Acts::BinUtility bu(15, -1.0, 1.0, Acts::open,
                              Acts::AxisDirection::AxisX);
          bu += Acts::BinUtility(15, -1.0, 1.0, Acts::open,
                                 Acts::AxisDirection::AxisY);

          sensorSurface->assignSurfaceMaterial(
              std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
        }
        // Helper: assign ProtoSurfaceMaterial based on bounds type.
        // Called for both front and back DSSD surfaces.
        auto assignSensorMaterial = [&](std::shared_ptr<Acts::Surface> surf) {
          if (surf->bounds().type() == Acts::SurfaceBounds::eConvexPolygon) {
            // Slanted sensors have ConvexPolygon bounds (TGeo trapezoid).
            // Derive bin ranges from the bounding box in local (X, Y).
            const auto* polyBounds =
                dynamic_cast<const Acts::ConvexPolygonBoundsBase*>(
                    &surf->bounds());
            if (polyBounds != nullptr) {
              const auto& bbox = polyBounds->boundingBox();
              Acts::BinUtility bu(s_factor * 50, bbox.min().x(), bbox.max().x(),
                                  Acts::open, Acts::AxisDirection::AxisX);
              bu += Acts::BinUtility(s_factor * 50, bbox.min().y(), bbox.max().y(),
                                     Acts::open, Acts::AxisDirection::AxisY);
              surf->assignSurfaceMaterial(
                  std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
              
              // Sanity check: all vertices must fall within the bin range
              for (const auto& vtx : polyBounds->vertices()) {
                  if (vtx.x() < bbox.min().x() || vtx.x() > bbox.max().x() ||
                      vtx.y() < bbox.min().y() || vtx.y() > bbox.max().y()) {
                      ACTS_ERROR("ConvexPolygon vertex (" << vtx.x() << ", " << vtx.y()
                                << ") lies OUTSIDE the bin grid!");
                  }
              }
              ACTS_INFO("ConvexPolygon bbox: x=[" << bbox.min().x() << ", " << bbox.max().x()
                        << "] y=[" << bbox.min().y() << ", " << bbox.max().y() << "]");
            }
            ACTS_INFO("Sensor: ConvexPolygon - assigned ProtoSurfaceMaterial");
          } else {
            // Barrel sensors: Rectangle / Trapezoid bounds.
            Acts::BinUtility bu(s_factor * 15, -1.0, 1.0, Acts::open,
                                Acts::AxisDirection::AxisX);
            bu += Acts::BinUtility(s_factor * 15, -1.0, 1.0, Acts::open,
                                   Acts::AxisDirection::AxisY);
            surf->assignSurfaceMaterial(
                std::make_shared<Acts::ProtoSurfaceMaterial>(bu));
            ACTS_INFO("Sensor: bounds type="
                      << static_cast<int>(surf->bounds().type())
                      << " - assigned ProtoSurfaceMaterial");
          }
        };

        assignSensorMaterial(tgElementFront->surface().getSharedPtr());

        sVector.push_back(tgElementFront->surface().getSharedPtr());

        if (layerCfg.isDSSD) {
          double gapCm = layerCfg.backsideGap / (1 * Acts::UnitConstants::cm);
          TGeoTranslation localShift(0, 0, gapCm);

          TGeoHMatrix backTransform(*snode.transform);
          backTransform.Multiply(&localShift);

          auto backIdentifier = identifier | 1;

          auto tgElementBack = m_cfg.detectorElementFactory(
              backIdentifier, *snode.node, backTransform,
              ActsPlugins::TGeoAxes("YxZ"), 1 * Acts::UnitConstants::cm,
              nullptr);

          m_elementStore.push_back(tgElementBack);
          auto backSurface = tgElementBack->surface().getSharedPtr();
          assignSensorMaterial(backSurface);
          sVector.push_back(backSurface);
        }
      }

      ACTS_DEBUG("- created TGeoDetectorElements : " << sVector.size());

      fillLayer(sVector, layerCfg);
    }
  }
  return cpLayers;
}
}  // namespace
}  // namespace B2

Belle2::Belle2(const Config& cfg)
    : Detector(Acts::getDefaultLogger("Belle2", cfg.logLevel)), m_cfg(cfg) {
  m_nominalGeometryContext =
      Acts::GeometryContext::dangerouslyDefaultConstruct();

  ACTS_INFO("Building tracking geometry");
  if (m_trackingGeometry != nullptr) {
    throw std::runtime_error("Tracking geometry already built");
  }

  B2::Belle2Builder::Config b_cfg;
  // Auch hier wurde Schabernack betrieben, diesmal mit der detectorelement
  // factory
  b_cfg.protoMaterial = m_cfg.buildProto;
  b_cfg.layerLogLevel = m_cfg.layerLogLevel;
  b_cfg.buildLevel = m_cfg.buildLevel;
  b_cfg.identifierProvider = std::make_shared<MyIdentifierProvider>();

  B2::Belle2Builder::LayerConfig cl1{
      .volumeName = "PXD.Envelope",
      .sensorNames = {"PXD.1.*.*.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {12, 16}}, // Define R and Z values
                      {Acts::AxisDirection::AxisZ, {-60, 80}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(7)},
      .binning1 = {std::variant<int, Acts::BinningType>(2)},
      .envelope = {1, 1},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl2{
      .volumeName = "PXD.Envelope",
      .sensorNames = {"PXD.2.*.*.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {20, 24}},
                      {Acts::AxisDirection::AxisZ, {-80, 100}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(7)},
      .binning1 = {std::variant<int, Acts::BinningType>(2)},
      .envelope = {1, 1},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};

  B2::Belle2Builder::LayerConfig cl3{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.3.*.*.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {41.4, 45.9}},
                      {Acts::AxisDirection::AxisZ, {-94.1, 153.8}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(7)},
      .binning1 = {std::variant<int, Acts::BinningType>(2)},
      .envelope = {30, 30},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl4{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.4.*.2.Active", "SVD.4.*.3.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {56.9, 89.4}},
                      {Acts::AxisDirection::AxisZ, {-161.7, 210.5}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(10)},
      .binning1 = {std::variant<int, Acts::BinningType>(2)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl5{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.5.*.2.Active", "SVD.5.*.3.Active",
                      "SVD.5.*.4.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {82.0, 111.6}},
                      {Acts::AxisDirection::AxisZ, {-200.9, 185.2}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(12)},
      .binning1 = {std::variant<int, Acts::BinningType>(3)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl6{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.6.*.2.Active", "SVD.6.*.3.Active",
                      "SVD.6.*.4.Active", "SVD.6.*.5.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {108, 143.6}},
                      {Acts::AxisDirection::AxisZ, {-254.4, 255.0}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(16)},
      .binning1 = {std::variant<int, Acts::BinningType>(4)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl4s{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.4.*.1.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {72.0, 110.0}},
                      {Acts::AxisDirection::AxisZ, {180.0, 300.0}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(10)},
      .binning1 = {std::variant<int, Acts::BinningType>(1)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl5s{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.5.*.1.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {72.0, 110.0}},
                      {Acts::AxisDirection::AxisZ, {180.0, 300.0}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(12)},
      .binning1 = {std::variant<int, Acts::BinningType>(1)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  B2::Belle2Builder::LayerConfig cl6s{
      .volumeName = "SVD.Envelope",
      .sensorNames = {"SVD.6.*.1.Active"},
      .parseRanges = {{Acts::AxisDirection::AxisR, {90.0, 143.0}},
                      {Acts::AxisDirection::AxisZ, {240.0, 360.0}}},
      .splitConfigs = {},
      .binning0 = {std::variant<int, Acts::BinningType>(16)},
      .binning1 = {std::variant<int, Acts::BinningType>(1)},
      .envelope = {5, 5},
      .localAxes = ActsPlugins::TGeoAxes("XYZ")};
  Acts::MaterialSlab svdSlab(B2::Belle2Builder::kSiliconMaterial, 0.32f);
  auto svdModuleMaterial =
      m_cfg.buildProto
          ? std::static_pointer_cast<const Acts::ISurfaceMaterial>(
                std::make_shared<Acts::ProtoSurfaceMaterial>())
          : std::static_pointer_cast<const Acts::ISurfaceMaterial>(
                std::make_shared<Acts::HomogeneousSurfaceMaterial>(svdSlab));

  cl3.moduleMaterial = svdModuleMaterial;
  cl4.moduleMaterial = svdModuleMaterial;
  cl5.moduleMaterial = svdModuleMaterial;
  cl6.moduleMaterial = svdModuleMaterial;
  cl4s.moduleMaterial = svdModuleMaterial;
  cl5s.moduleMaterial = svdModuleMaterial;
  cl6s.moduleMaterial = svdModuleMaterial;
  cl3.isDSSD = true;
  cl3.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl3.backsideStereo = M_PI / 2.0;

  cl4.isDSSD = true;
  cl4.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl4.backsideStereo = M_PI / 2.0;
  cl5.isDSSD = true;
  cl5.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl5.backsideStereo = M_PI / 2.0;
  cl6.isDSSD = true;
  cl6.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl6.backsideStereo = M_PI / 2.0;
  cl6s.isDSSD = true;
  cl6s.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl6s.backsideStereo = M_PI / 2.0;
  cl4s.isDSSD = true;
  cl4s.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl4s.backsideStereo = M_PI / 2.0;
  cl5s.isDSSD = true;
  cl5s.backsideGap = 0.32 * Acts::UnitConstants::mm;
  cl5s.backsideStereo = M_PI / 2.0;
  b_cfg.pxd_layerConfigs.push_back(cl1);
  b_cfg.pxd_layerConfigs.push_back(cl2);
  b_cfg.layerConfigs.push_back(cl3);
  b_cfg.layerConfigs.push_back(cl4);
  b_cfg.layerConfigs.push_back(cl5);
  b_cfg.layerConfigs.push_back(cl6);
  b_cfg.layerConfigs.push_back(cl4s);
  b_cfg.layerConfigs.push_back(cl5s);
  b_cfg.layerConfigs.push_back(cl6s);
  B2::Gen3Belle2Builder builder(b_cfg, logger().clone());

  m_trackingGeometry = builder.buildTrackingGeometry(m_nominalGeometryContext,
                                                     m_cfg.graphvizFile);

  if (m_trackingGeometry == nullptr) {
    throw std::runtime_error("Error building tracking geometry");
  }
  if (m_cfg.materialDecorator) {
    ACTS_INFO("Wende MaterialDecorator auf die gebaute Geometrie an...");
    m_trackingGeometry->visitSurfaces([&](const Acts::Surface* surface) {
      if (surface) {
        m_cfg.materialDecorator->decorate(const_cast<Acts::Surface&>(*surface));
      }
    }, false);
    m_trackingGeometry->visitVolumes([&](const Acts::TrackingVolume* volume) {
      if (volume) {
        m_cfg.materialDecorator->decorate(
            const_cast<Acts::TrackingVolume&>(*volume));

        for (const auto& boundary : volume->boundarySurfaces()) {
          if (boundary) {
            const Acts::Surface& portalSurface =
                boundary->surfaceRepresentation();
            m_cfg.materialDecorator->decorate(
                const_cast<Acts::Surface&>(portalSurface));
          }
        }
      }
    });
  }

  auto elements = builder.getElementStore();
  m_detectorStore.insert(m_detectorStore.end(), elements.begin(),
                         elements.end());
  auto cdcElements = builder.getCDCElementStore();
  m_detectorStore.insert(m_detectorStore.end(), cdcElements.begin(),
                         cdcElements.end());

  ACTS_INFO("Tracking geometry built");
}

}  // namespace ActsExamples
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsExamples/Belle2/CDCWireElement.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include <map>


struct MyIdentifierHook: public Acts::GeometryIdentifierHook {

    // Maps VXD layer number to the volume ID used in the CSV encoding
    // (must match LAYER_TO_ADDRESS in finding_basf2_cluster_true.py)
    // PXD: layers 1-2 → volumes 15-16; SVD: layers 3-6 → volumes 17-20
    static constexpr std::array<std::pair<int,int>, 6> kLayerToVolume = {{
        {1, 15}, {2, 16}, {3, 17}, {4, 18}, {5, 19}, {6, 20}
    }};

    Acts::GeometryIdentifier decorateIdentifier(
        Acts::GeometryIdentifier identifier,
        const Acts::Surface& surface) const override {

        const Acts::SurfacePlacementBase* detElem = surface.surfacePlacement();
        if (detElem) {
            // CDC wires: restore the pre-set identifier (volume=14, layer=ICLayer+1,
            // sensitive=IWire+1) that Blueprint overwrites with sequential IDs.
            auto cdcElem = dynamic_cast<const CDCWireElement*>(detElem);
            if (cdcElem) {
                return cdcElem->identifier();
            }

            // VXD (PXD/SVD) sensors: set volume from LAYER_TO_ADDRESS, sensitive = VxdID.
            auto tgeoElem = dynamic_cast<const ActsPlugins::TGeoDetectorElement*>(detElem);
            if (tgeoElem) {
                auto vxdId = static_cast<Acts::GeometryIdentifier::Value>(tgeoElem->identifier());
                int vxdLayer = (vxdId >> 13) & 0x7;
                for (auto [layer, vol] : kLayerToVolume) {
                    if (layer == vxdLayer) {
                        return Acts::GeometryIdentifier{}
                            .withVolume(vol)
                            .withSensitive(vxdId);
                    }
                }
            }
        }
        return identifier;
    }
};
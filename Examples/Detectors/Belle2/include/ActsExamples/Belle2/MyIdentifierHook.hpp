#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include <map>


struct MyIdentifierHook: public Acts::GeometryIdentifierHook {

    // Maps SVD layer number to the volume ID used in the CSV encoding
    // (must match LAYER_TO_ADDRESS in finding_basf2_cluster_true.py)
    static constexpr std::array<std::pair<int,int>, 4> kLayerToVolume = {{
        {3, 17}, {4, 18}, {5, 19}, {6, 20}
    }};

    Acts::GeometryIdentifier decorateIdentifier(
        Acts::GeometryIdentifier identifier,
        const Acts::Surface& surface) const override {

        const Acts::SurfacePlacementBase* detElem = surface.surfacePlacement();
        if (detElem) {
            auto tgeoElem = dynamic_cast<const ActsPlugins::TGeoDetectorElement*>(detElem);
            if (tgeoElem) {
                // tgeoID is the VxdID-encoded value: (layer<<13)|(ladder<<8)|(sensor<<5)
                // set by MyIdentifierProvider::identify()
                auto vxdId = static_cast<Acts::GeometryIdentifier::Value>(tgeoElem->identifier());
                int svdLayer = (vxdId >> 13) & 0x7;
                for (auto [layer, vol] : kLayerToVolume) {
                    if (layer == svdLayer) {
                        // Build the full geometry ID to match the CSV encoding:
                        // volume from LAYER_TO_ADDRESS, sensitive = VxdID value
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
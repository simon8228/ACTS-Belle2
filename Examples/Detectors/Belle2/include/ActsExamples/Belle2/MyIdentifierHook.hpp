#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include <iostream>


struct MyIdentifierHook: public Acts::GeometryIdentifierHook {

    Acts::GeometryIdentifier decorateIdentifier(
        Acts::GeometryIdentifier identifier,
        const Acts::Surface& surface) const override {
            
            const Acts::SurfacePlacementBase* detElem = surface.surfacePlacement();
            Acts::GeometryIdentifier newId = identifier;
            if(detElem){
                auto tgeoElem = dynamic_cast<const ActsPlugins::TGeoDetectorElement*>(detElem);
                if(tgeoElem){
                    auto tgeoID = tgeoElem->identifier();
                    newId = identifier.withSensitive(static_cast<Acts::GeometryIdentifier::Value>(tgeoID));
            }
        }
        return newId;
    }
};
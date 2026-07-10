#pragma once

#include "ActsPlugins/Root/ITGeoIdentifierProvider.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"

#include "Acts/Geometry/GeometryContext.hpp"
#include <string>
#include "TGeoNode.h"

class MyIdentifierProvider : public ActsPlugins::ITGeoIdentifierProvider {
  public:
	MyIdentifierProvider() = default;
	~MyIdentifierProvider() override = default;

	ActsPlugins::TGeoDetectorElement::Identifier identify(
	    const Acts::GeometryContext& gctx, const TGeoNode& tgnode) const override {

		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream(tgnode.GetName());
		while (std::getline(tokenStream, token, '.')) {
			tokens.push_back(token);
		}
		tokens.erase(tokens.begin());
		tokens.erase(tokens.end() - 1);

		int layer = std::stoi(tokens[0]);
		int ladder = std::stoi(tokens[1]);
		int sensor = std::stoi(tokens[2]);
		int id = (layer << 13) | (ladder << 8) | (sensor << 5);

		return static_cast<ActsPlugins::TGeoDetectorElement::Identifier>(tgnode.GetNumber());
	}
};
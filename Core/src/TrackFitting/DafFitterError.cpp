// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/TrackFitting/DafFitterError.hpp"

#include <string>

namespace {

class DafFitterErrorCategory : public std::error_category {
 public:
  // Return a short descriptive name for the category.
  const char* name() const noexcept final { return "DafFitterError"; }

  // Return what each enum means in text.
  std::string message(int c) const final {
    using Acts::DafFitterError;

    switch (static_cast<DafFitterError>(c)) {
      case DafFitterError::UpdateFailed:
        return "Kalman update of the effective measurement failed";
      case DafFitterError::SmoothFailed:
        return "Smoothing failed";
      case DafFitterError::OutputConversionFailed:
        return "Output conversion failed";
      case DafFitterError::NoMeasurementFound:
        return "No measurement group detected during the propagation";
      case DafFitterError::InconsistentCandidateGroup:
        return "A candidate group contained source links resolving to more "
               "than one surface";
      case DafFitterError::AnnealingDidNotConverge:
        return "The annealing loop did not reach the lowest configured "
               "temperature";
      case DafFitterError::PropagationStalled:
        return "Propagation made no net forward progress for an extended "
               "number of steps and was aborted";
      default:
        return "unknown";
    }
  }
};

}  // namespace

std::error_code Acts::make_error_code(Acts::DafFitterError e) {
  static DafFitterErrorCategory c;
  return {static_cast<int>(e), c};
}

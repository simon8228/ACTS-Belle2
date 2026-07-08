// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <system_error>
#include <type_traits>

namespace Acts {

/// Error codes for Deterministic Annealing Filter operations
/// @ingroup errors
enum class DafFitterError {
  // ensure all values are non-zero
  /// Kalman update of the effective measurement failed
  UpdateFailed = 1,
  /// Smoothing failed
  SmoothFailed,
  /// Output conversion failed
  OutputConversionFailed,
  /// No measurement group detected during the propagation
  NoMeasurementFound,
  /// A candidate group contained source links resolving to more than one
  /// surface
  InconsistentCandidateGroup,
  /// The annealing loop did not reach the lowest configured temperature
  AnnealingDidNotConverge,
  /// Propagation made no net forward progress for an extended number of
  /// steps (e.g. an oscillating, non-converging intersection search for the
  /// next target surface), and was aborted rather than exhausting the full
  /// step budget
  PropagationStalled,
};

/// Create error code from @ref DafFitterError
/// @param e The error code enum value
/// @return Standard error code
std::error_code make_error_code(Acts::DafFitterError e);

}  // namespace Acts

namespace std {
// register with STL
template <>
struct is_error_code_enum<Acts::DafFitterError> : std::true_type {};
}  // namespace std

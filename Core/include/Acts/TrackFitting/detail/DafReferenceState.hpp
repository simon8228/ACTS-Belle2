// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Surfaces/Surface.hpp"

#include <cstddef>
#include <vector>

namespace Acts {

/// One entry of a DAF reference trajectory -- the ACTS analogue of GenFit's
/// @c genfit::ReferenceStateOnPlane.
///
/// The DAF's default filter linearises about its own running filtered state:
/// the propagator transports the estimate, and every prediction, Jacobian and
/// measurement calibration is evaluated wherever that estimate currently sits
/// (see DafActor, which pushes the filtered state back into the stepper). That
/// is structurally GenFit's plain @c KalmanFitter.
///
/// GenFit's DAF instead runs @c KalmanFitterRefTrack: a preparation pass
/// extrapolates one reference trajectory through every surface and stores, per
/// surface, the reference parameters, the transport Jacobian F, the process
/// noise N and the delta state c. The filter then never propagates -- it
/// applies the stored affine map
///
///     p_k|k-1 = F_k * p_k-1|k-1 + c_k
///     C_k|k-1 = F_k * C_k-1|k-1 * F_k^T + N_k
///
/// so all hits in a pass share one linearisation point, and the measurements
/// are calibrated on that same reference rather than on a partially-informed
/// running estimate. This struct is the stored record that makes that possible.
///
/// One entry is produced per surface the reference propagation touches,
/// measurement surfaces and material/hole surfaces alike, in propagation
/// order: the Jacobian chain runs through the material surfaces too, so they
/// have to take part in the affine prediction even though nothing is updated
/// there.
struct DafReferenceState {
  /// The surface this state sits on.
  const Surface* surface = nullptr;

  /// Reference parameters p_ref,k on that surface. This is the linearisation
  /// point: predictions are expressed as deviations from it, and the
  /// candidate measurements were calibrated against it.
  BoundVector parameters = BoundVector::Zero();

  /// Transport Jacobian F_k from the PREVIOUS reference state to this one.
  /// Identity for the first state.
  BoundMatrix jacobian = BoundMatrix::Identity();

  /// Process noise N_k accumulated over the segment ending at this surface
  /// (the previous surface's post-update material, the material crossed along
  /// the way, and this surface's pre-update material).
  BoundSquareMatrix noise = BoundSquareMatrix::Zero();

  /// Delta state c_k = p_ref,k - F_k * p_ref,k-1, i.e. the constant that makes
  /// the affine model reproduce the reference transport exactly. Defined
  /// exactly as GenFit's @c RKTrackRep::getForwardJacobianAndNoise computes it.
  ///
  /// Recorded for reference and for tests; detail::dafReferenceFilter predicts
  /// with the algebraically identical deviation form p_ref,k + F*(p - p_ref,k-1)
  /// instead, which unlike p = F*p + c survives the phi/theta branch cut.
  BoundVector deltaState = BoundVector::Zero();

  /// Accumulated path length at this surface, as reported by the reference
  /// propagation.
  double pathLength = 0.;

  /// Whether this surface carries a measurement group.
  bool hasMeasurement = false;

  /// Whether this surface has material.
  bool hasMaterial = false;

  /// Whether this surface is a hole (sensitive, no measurement, and a
  /// measurement precedes it).
  bool isHole = false;

  /// Indices, into the persistent candidate store, of this surface's
  /// calibrated candidate track states -- in the same order as the surface's
  /// DafMeasurementGroup and therefore as its weights. Empty unless
  /// @c hasMeasurement.
  std::vector<std::size_t> candidateIndices;
};

/// A full reference trajectory, in propagation order.
using DafReferenceTrajectory = std::vector<DafReferenceState>;

}  // namespace Acts

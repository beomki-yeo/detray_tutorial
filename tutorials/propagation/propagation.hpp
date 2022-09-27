/** Detray tutorial project, No copy right **/

#pragma once

// Detray include(s).
#include "detray/plugins/algebra/array_definitions.hpp"
#include "detray/definitions/units.hpp"
#include "detray/field/constant_magnetic_field.hpp"
#include "detray/propagator/aborters.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/propagator/track.hpp"
#include "tests/common/tools/create_toy_geometry.hpp"
#include "tests/common/tools/track_generators.hpp"

using namespace detray;

// TrackML (toy) detector type for host and device.
// The device detector is defined with vecmem::device_vector 
// and vecmem::jagged_device_vector.
using detector_host_type =
    detector<detector_registry::toy_detector, std::array, thrust::tuple,
             vecmem::vector, vecmem::jagged_vector>;
using detector_device_type =
    detector<detector_registry::toy_detector, std::array, thrust::tuple,
             vecmem::device_vector, vecmem::jagged_device_vector>;

// Detector setup for the number of layers
constexpr std::size_t n_barrel_layers = 4;
constexpr std::size_t n_endcap_layers = 7;

// Navigator type for host and device
using navigator_host_type = navigator<detector_host_type>;
using navigator_device_type = navigator<detector_device_type>;

// Runge-Kutta stepper with constant magnetic field and default constrained step.
using rk_stepper_type =
    rk_stepper<constant_magnetic_field<>, free_track_parameters, constrained_step<>>;

// Propagator type for host and device
using propagator_host_type =
    propagator<rk_stepper_type, navigator_host_type, actor_chain<>>;
using propagator_device_type =
    propagator<rk_stepper_type, navigator_device_type, actor_chain<>>;

// CUDA propagation function
void cuda_propagation(
    detector_view<detector_host_type> det_data,
    const constant_magnetic_field<> B,
    vecmem::data::vector_view<free_track_parameters> &tracks_data,
    vecmem::data::jagged_vector_view<line_plane_intersection> &candidates_data);

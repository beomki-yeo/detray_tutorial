/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Choose an algebra-plugin
#include "detray/plugins/algebra/array_definitions.hpp"

// detray includes
#include "detray/definitions/units.hpp"
#include "detray/field/constant_magnetic_field.hpp"
#include "detray/intersection/detail/trajectories.hpp" // helix
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/propagator/track.hpp"
#include "tests/common/tools/create_toy_geometry.hpp"
#include "tests/common/tools/inspectors.hpp"
#include "tests/common/tools/particle_gun.hpp"
#include "tests/common/tools/track_generators.hpp"

// vecmem includes
#include <vecmem/memory/host_memory_resource.hpp>

#include <sstream>

using namespace detray;
using namespace detray::navigation;

using object_tracer_t =
    object_tracer<dvector, status::e_on_module, status::e_on_portal>;
using inspector_t = aggregate_inspector<object_tracer_t, print_inspector>;

// Build the toy detector and run intersections with 
int main()
{
    // Detector configuration (this particular one is the most well tested)
    constexpr std::size_t n_brl_layers{4}; // up to 4 barrel layers
    constexpr std::size_t n_edc_layers{1}; // up to 7 endacap layers
    // Do host-side allocation only
    vecmem::host_memory_resource host_mr;

    // Pixel detector of the ACTS generic detector
    auto det = create_toy_geometry(host_mr, n_brl_layers, n_edc_layers);

    // Runge-Kutta based navigation
    using navigator_t = navigator<decltype(det), inspector_t>;

    using stepper_t = rk_stepper<constant_magnetic_field<>, 
                                 free_track_parameters>;
    using propagator_t = propagator<stepper_t, navigator_t, actor_chain<>>;

    const vector3 B{0. * unit_constants::T,
                    0. * unit_constants::T,
                    2. * unit_constants::T};
    constant_magnetic_field<> b_field(B);
    propagator_t prop(stepper_t{b_field}, navigator_t{det});

    // Track generation config
    // Trivial example: Single track escapes through beampipe
    constexpr std::size_t theta_steps{1};
    constexpr std::size_t phi_steps{1};

    // det.volume_by_pos(ori).index();
    const point3 ori{0., 0., 0.};
    constexpr scalar p_mag{10. * unit_constants::GeV};

    // Iterate through uniformly distributed momentum directions
    for (auto track : uniform_track_generator<free_track_parameters>(
             theta_steps, phi_steps, ori, p_mag)) {
        track.set_overstep_tolerance(-7. * unit_constants::um);

        // Get ground truth helix from track
        detail::helix helix(track, &B);

        // Shoot helix through the detector and record all surface intersections
        const auto intersection_trace =
            particle_gun::shoot_particle(det, helix);

        // Now follow that helix with the same track and check, if we find
        // the same volumes and distances along the way
        propagator_t::state propagation(track);

        // Retrieve navigation information
        auto &inspector = propagation._navigation.inspector();
        auto &obj_tracer = inspector.template get<object_tracer_t>();
        auto &debug_printer = inspector.template get<print_inspector>();

        // Run the actual propagation
        prop.propagate(propagation);
        std::cout << debug_printer.to_string();

        // Compare helix trace to object tracer
        std::stringstream debug_stream;
        for (std::size_t intr_idx = 0; intr_idx < intersection_trace.size();
             ++intr_idx) {
            debug_stream << "-------Intersection trace\n"
                         << "helix gun: "
                         << "\tvol id: " << intersection_trace[intr_idx].first
                         << ", "
                         << intersection_trace[intr_idx].second.to_string();
            debug_stream << "navig.: " << obj_tracer[intr_idx].to_string();
        }

        // Compare intersection records
        //
        // [...]
        //
    }
}

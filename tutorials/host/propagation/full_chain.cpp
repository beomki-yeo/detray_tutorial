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
#include "detray/propagator/aborters.hpp"
#include "detray/propagator/actor_chain.hpp"
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

    // Define navigator, stepper, actor chain and propagator
    using navigator_t = navigator<decltype(det)>;

    using b_field_t = constant_magnetic_field<>;
    using track_t = free_track_parameters;
    using constraints_t = constrained_step<>; // different step-size constr.
    using policy_t = stepper_default_policy; // how to update the navigation
    using stepper_t = rk_stepper<b_field_t, track_t, constraints_t, policy_t>;

    using actor_chain_t =
        actor_chain<dtuple, propagation::print_inspector, pathlimit_aborter>;

    using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

    // Constant magnetic field
    const auto B = __plugin::vector3<scalar>{0. * unit_constants::T,
                                             0. * unit_constants::T,
                                             2. * unit_constants::T};
    const b_field_t b_field(B);

    // Propagator is built from the stepper and navigator
    propagator_t p(stepper_t{b_field}, navigator_t{det});

    // Track generation config
    // Trivial example: Single track escapes through beampipe
    constexpr std::size_t theta_steps{10};
    constexpr std::size_t phi_steps{10};

    // det.volume_by_pos(ori).index();
    const point3 ori{0., 0., 0.};
    constexpr scalar p_mag{10. * unit_constants::GeV};

    // Test parameters
    constexpr scalar epsilon{1e-3};
    constexpr scalar overstep_tol{-7. * unit_constants::um};
    constexpr scalar step_constr{30 * unit_constants::cm};
    constexpr scalar path_limit{60 * unit_constants::cm};

    // Iterate through uniformly distributed momentum directions
    for (auto track :
         uniform_track_generator<track_t>(theta_steps, phi_steps, ori, p_mag)) {

        // Define opverstepping tolerance
        track.set_overstep_tolerance(overstep_tol);

        // Build actor states and tie them together
        propagation::print_inspector::state print_insp_state{};
        pathlimit_aborter::state pathlimit_aborter_state{path_limit};

        actor_chain_t::state actor_states = std::tie(
            print_insp_state, pathlimit_aborter_state);

        // Init propagator state
        propagator_t::state p_state(track, actor_states);

        // Set step constraints (the most strict will be applied)
        p_state._stepping
            .template set_constraint<step::constraint::e_accuracy>(step_constr);

        // Propagate the track
        bool is_success = p.propagate(p_state);

        // Check
        std::cout << "===============================" << std::endl;
        if (not is_success and p_state._stepping.path_length() <
                    path_limit + epsilon) {
            std::cout << "aborted: reached path limit" << std::endl;
        }
        else if (not is_success) {
            std::cout << "unknown error: " << std::endl;
            std::cout << print_insp_state.to_string() << std::endl;
        }
        else {
            std::cout << "successful propagation" << std::endl;
        }
    }
}

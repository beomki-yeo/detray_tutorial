/** Detray tutorial project, No copy right **/

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

// Vecmem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace detray;

/// Actor definition to inspect a track state on surfaces
struct track_inspector : actor
{
    struct state
    {
    };

    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(
        state & /*inspector_state*/, const propagator_state_t &prop_state) const
    {

        const auto &stepping = prop_state._stepping;
        const auto &navigation = prop_state._navigation;

        // Print out the track state when the track is on module (physical surface)
        if (navigation.is_on_module())
        {
            const auto &track = stepping();
            const auto pos = track.pos();
            const auto dir = track.dir();

            std::size_t volume_id = navigation.current()->link;
            std::size_t surface_id = navigation.current()->index;

            std::cout << "Volume ID:" << std::setw(3) << volume_id << "   ";
            std::cout << "Surface ID:" << std::setw(5) << surface_id << "   ";

            std::stringstream position, direction;
            position << "(" << pos[0] << "," << pos[1] << "," << pos[2] << ")";
            direction << "(" << dir[0] << "," << dir[1] << "," << dir[2] << ")";

            std::cout << "Position:" << std::setw(25) << position.str() << "   ";
            std::cout << "Direction:" << std::setw(25) << direction.str() << "   ";
            std::cout << std::endl;
        }
    }
};

/********************
 * Type definitions *
 ********************/

using detector_type =
    detector<detector_registry::toy_detector, std::array, std::tuple,
             vecmem::vector, vecmem::jagged_vector>;

using navigator_type = navigator<detector_type>;

using rk_stepper_type =
    rk_stepper<constant_magnetic_field<>, free_track_parameters,
               constrained_step<>>;

using actor_chain_type = actor_chain<std::tuple, track_inspector>;

using propagator_type =
    propagator<rk_stepper_type, navigator_type, actor_chain_type>;

// Main
int main()
{

    /*****************
     * Initial Setup *
     *****************/

    // Detector setup for the number of layers
    constexpr std::size_t n_barrel_layers = 4;
    constexpr std::size_t n_endcap_layers = 7;

    // Set up the constant magnetic field
    const vector3 B{0, 0, 2 * unit_constants::T};
    constant_magnetic_field<> B_field(B);

    // VecMem memory resource(s)
    vecmem::host_memory_resource host_resource;

    // Create a track
    const vector3 pos{0., 0., 0};                // Position at origin
    const vector3 dir{0., 1., 0};                // Direction into y-axis
    const scalar mom = 10 * unit_constants::GeV; // 10 GeV momentum
    const scalar time = 0;                       // Time
    const scalar charge = -1.;                   // Negative charge
    const free_track_parameters track(pos, time, mom * dir, charge);

    /***************
     * propagation *
     ***************/

    // Create the TrackML (toy) geometry
    const detector_type detector =
        create_toy_geometry<std::array, std::tuple, vecmem::vector,
                            vecmem::jagged_vector>(host_resource,
                                                   n_barrel_layers,
                                                   n_endcap_layers);

    // Create RK stepper
    rk_stepper_type s(B_field);

    // Create navigator
    navigator_type n(detector);

    // Create propagator
    propagator_type p(std::move(s), std::move(n));

    // Actor state
    track_inspector::state insp_state{};

    // Create the propagator state
    propagator_type::state state(track, insp_state);

    // Run propagation
    p.propagate(state);

    return 0;
}
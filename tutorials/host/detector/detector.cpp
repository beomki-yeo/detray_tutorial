/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <iostream>

// Choose an algebra-plugin
#include "detray/plugins/algebra/array_definitions.hpp"

// detray includes
#include "detray/intersection/intersection_kernel.hpp"
#include "detray/intersection/detail/trajectories.hpp" // ray
#include "tests/common/tools/create_toy_geometry.hpp"
#include "tests/common/tools/particle_gun.hpp"
#include "tests/common/tools/ray_scan_utils.hpp"
#include "tests/common/tools/track_generators.hpp"

// vecmem includes
#include <vecmem/memory/host_memory_resource.hpp>


using namespace detray;

// Build the toy detector and run intersections with 
int main()
{
    // Detector configuration (this particular one is the most well tested)
    constexpr std::size_t n_brl_layers{4}; // up to 4 barrel layers
    constexpr std::size_t n_edc_layers{3}; // up to 7 endacap layers
    // Do host-side allocation only
    vecmem::host_memory_resource host_mr;

    // Pixel detector of the ACTS generic detector
    auto det = create_toy_geometry(host_mr, n_brl_layers, n_edc_layers);

    // Track generation config
    constexpr std::size_t theta_steps{10};
    constexpr std::size_t phi_steps{10};
    const point3 ori{0., 0., 0.};

    unsigned int hits{0};
    unsigned int missed{0};

    bool is_consisten_linking = true;

    // Iterate through uniformly distributed momentum directions
    for (const auto ray :
         uniform_track_generator<detail::ray>(theta_steps, phi_steps, ori)) {

        // Loop over volumes
        for (const auto &v : det.volumes()) {
            // Loop over all surfaces in volume
            for (const auto &sf : range(det.surfaces(), v)) {

                auto sfi =
                    det.mask_store().template execute<intersection_update>(
                        sf.mask_type(), ray, sf, det.transform_store());

                sfi.status == intersection::status::e_inside ? ++hits : ++missed;
            }
        }

        //
        // ray scan (also possible with helix trajectory)
        //
        // Shoot ray through the detector and record all surfaces it encounters
        const auto intersection_record =
            particle_gun::shoot_particle(det, ray);  // :)

        // Create a trace of the volume indices that were encountered
        dindex start_index{0};
        auto [portal_trace, surface_trace] =
            trace_intersections(intersection_record, start_index);

        // Check correct portal linking
        is_consisten_linking &= check_connectivity(portal_trace);
    }

    std::cout << "[detray] hits / missed / total = " << hits << " / " << missed
              << " / " << hits + missed << std::endl;

    // Is this a sensible trace to be further examined?
    std::cout << "Detector has consistent linking: "
                << std::boolalpha << is_consisten_linking << std::endl;
}

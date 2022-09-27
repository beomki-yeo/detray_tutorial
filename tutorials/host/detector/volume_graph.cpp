/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Choose an algebra-plugin
#include "detray/plugins/algebra/array_definitions.hpp"

// detray includes
#include "detray/geometry/volume_graph.hpp"
#include "tests/common/tools/create_toy_geometry.hpp"
#include "tests/common/tools/hash_tree.hpp"

// vecmem includes
#include <vecmem/memory/host_memory_resource.hpp>

#include <iostream>

using namespace detray;

// Build the toy detector and print the volume graph
int main()
{
    // Detector configuration (this particular one is the most well tested)
    constexpr std::size_t n_brl_layers{4}; // up to 4 barrel layers
    constexpr std::size_t n_edc_layers{3}; // up to 7 endacap layers
    // Do host-side allocation only
    vecmem::host_memory_resource host_mr;

    // Pixel detector of the ACTS generic detector
    auto det = create_toy_geometry(host_mr, n_brl_layers, n_edc_layers);

    // Print detector
    volume_graph graph(det);

    std::cout << graph.to_string() << std::endl;

    const auto &adj_mat = graph.adjacency_matrix();
    // Still WIP...
    auto geo_checker = hash_tree(adj_mat);
}

/** Detray tutorial project, No copy right **/

// Project include(s).
#include "propagation_cuda.hpp"

// Vecmem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Create a batch of tracks
void create_tracks(vecmem::vector<free_track_parameters> &tracks,
                   const unsigned int theta_steps, const unsigned int phi_steps)
{
    // Set origin position of tracks
    const point3 ori{0., 0., 0.};
    const scalar mom_mag = 10. * unit_constants::GeV;

    // Iterate through uniformly distributed momentum directions
    for (auto traj : uniform_track_generator<free_track_parameters>(
             theta_steps, phi_steps, ori, mom_mag))
    {
        tracks.push_back(traj);
    }
}

// Main
int main()
{
    /*****************
     * Initial Setup *
     *****************/

    // Track batch setup (10 X 10 == 100 tracks)
    constexpr int n_theta_steps = 10;
    constexpr int n_phi_steps = 10;

    // Detector setup for the number of layers
    constexpr std::size_t n_barrel_layers = 4;
    constexpr std::size_t n_endcap_layers = 7;

    // Set up the constant magnetic field
    const vector3 B{0, 0, 2 * unit_constants::T};
    constant_magnetic_field<> B_field(B);

    // VecMem memory resource(s)
    vecmem::cuda::managed_memory_resource managed_resource;
    vecmem::cuda::device_memory_resource device_resource;

    /*******************
     * CPU propagation *
     *******************/

    // Create the TrackML (toy) geometry for CPU
    detector_host_type detector =
        create_toy_geometry<std::array, thrust::tuple, vecmem::vector,
                            vecmem::jagged_vector>(managed_resource,
                                                   n_barrel_layers,
                                                   n_endcap_layers);

    // Create a batch of tracks for CPU
    vecmem::vector<free_track_parameters> tracks_host(&managed_resource);
    create_tracks(tracks_host, n_theta_steps, n_phi_steps);

    // Create RK stepper
    rk_stepper_type s(B_field);

    // Create navigator
    navigator_host_type n(detector);

    // Create propagator
    propagator_host_type propagator(std::move(s), std::move(n));

    // Do the propagation
    for (auto &track : tracks_host)
    {
        // Create the propagator state
        propagator_host_type::state state(track);

        // Run propagation
        propagator.propagate(state);
    }

    /********************
     * CUDA propagation *
     ********************/

    // Get detector data for memory transfer to the GPU
    auto det_data = get_data(detector);

    // Create a batch of tracks for CUDA
    vecmem::vector<free_track_parameters> tracks_device(&managed_resource);
    create_tracks(tracks_device, n_theta_steps, n_phi_steps);

    // Get tracks data for memory transfer to the GPU
    auto tracks_data = vecmem::get_data(tracks_device);

    // Create navigator candidates buffer
    auto candidates_buffer =
        create_candidates_buffer(detector, tracks_device.size(), device_resource);
    vecmem::cuda::copy copy;
    copy.setup(candidates_buffer);

    // Do the propagation in CUDA
    cuda_propagation(det_data, B_field, tracks_data, candidates_buffer);

    return 0;
}
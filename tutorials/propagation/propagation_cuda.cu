/** Detray tutorial project, No copy right **/

// Project include(s).
#include "propagation_cuda.hpp"

// Detray include(s).
#include "detray/definitions/cuda_definitions.hpp"

__global__ void cuda_propagation_kernel(
    detector_view<detector_host_type> det_data,
    const constant_magnetic_field<> B_field,
    vecmem::data::vector_view<free_track_parameters> tracks_data,
    vecmem::data::jagged_vector_view<intersection_t> candidates_data)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    detector_device_type det(det_data);
    vecmem::device_vector<free_track_parameters> tracks(tracks_data);
    vecmem::jagged_device_vector<intersection_t> candidates(candidates_data);

    if (gid >= tracks.size())
    {
        return;
    }

    // Create RK stepper
    rk_stepper_type s(B_field);

    // Create navigator
    navigator_device_type n(det);

    // Create propagator
    propagator_device_type propagator(std::move(s), std::move(n));

    // Create the propagator state
    propagator_device_type::state state(
        tracks.at(gid), actor_chain<>::state{}, candidates.at(gid));

    // Run propagation
    propagator.propagate(state);
}

// CUDA propagation function
void cuda_propagation(
    detector_view<detector_host_type> det_data,
    const constant_magnetic_field<> B,
    vecmem::data::vector_view<free_track_parameters> &tracks_data,
    vecmem::data::jagged_vector_view<intersection_t> &candidates_data)
{

    constexpr int thread_dim = 2 * WARP_SIZE;
    int block_dim = tracks_data.size() / thread_dim + 1;

    // run the test kernel
    cuda_propagation_kernel<<<block_dim, thread_dim>>>(
        det_data, B, tracks_data, candidates_data);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
# C++17 support for CUDA requires CMake 3.18.
cmake_minimum_required( VERSION 3.18 )

# Enable CUDA as a language.
enable_language( CUDA )

# propagator executable
detray_add_executable( tutorial_propagator_cpu
   "propagation/propagation_cpu.cpp" 
   LINK_LIBRARIES detray::array detray_tests_common detray::core vecmem::core)

# propagator executable
detray_add_executable( tutorial_propagator_cuda
   "propagation/propagation_cuda.cpp" "propagation/propagation_cuda.hpp" 
   "propagation/propagation_cuda.cu" 
   LINK_LIBRARIES detray::array detray_tests_common detray::core vecmem::cuda )

detray_add_flag( CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" )   
# C++17 support for CUDA requires CMake 3.18.
cmake_minimum_required( VERSION 3.18 )

# Enable CUDA as a language.
enable_language( CUDA )

# propagator executable
detray_add_executable( tutorial_propagator
   "propagation/propagation.cpp" "propagation/propagation.hpp" 
   "propagation/propagation.cu" 
   LINK_LIBRARIES detray::array detray_tests_common detray::core vecmem::cuda )

detray_add_flag( CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" )   
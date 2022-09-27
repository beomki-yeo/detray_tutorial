# C++17 support for CUDA requires CMake 3.18.
cmake_minimum_required( VERSION 3.18 )

detray_add_executable( tutorial_detector
   "host/detector/detector.cpp"
   LINK_LIBRARIES detray::array detray_tests_common detray::core )

detray_add_executable( tutorial_volume_graph
   "host/detector/volume_graph.cpp"
   LINK_LIBRARIES detray::array detray_tests_common detray::core )

detray_add_executable( tutorial_actors
   "host/propagation/actors.cpp"
   LINK_LIBRARIES detray::array detray_tests_common detray::core )

detray_add_executable( tutorial_navigation
   "host/propagation/navigation.cpp"
   LINK_LIBRARIES detray::array detray_tests_common detray::core )

detray_add_executable( tutorial_full_chain
   "host/propagation/full_chain.cpp"
   LINK_LIBRARIES detray::array detray_tests_common detray::core )

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
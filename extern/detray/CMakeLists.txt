# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building Detray" )

# Declare where to get Detray from.
set( DETRAY_SOURCE
   "URL;https://github.com/acts-project/detray/archive/refs/tags/v0.11.0.tar.gz;URL_MD5;da8b6c15eb990eba355dace69071dae9"
   CACHE STRING "Source for Detray, when built as part of this project" )
mark_as_advanced( DETRAY_SOURCE )
FetchContent_Declare( Detray ${DETRAY_SOURCE} )

# Options used in the build of Detray.
set( DETRAY_CUSTOM_SCALARTYPE "float" CACHE STRING
   "Scalar type to use in the Detray code" )
set( DETRAY_BUILD_TESTING TRUE CACHE BOOL
   "Turn on the build of the Detray unit tests" )
set( DETRAY_EIGEN_PLUGIN FALSE CACHE BOOL
   "Turn off the build of the Detray Eigen code" )
set( DETRAY_VC_PLUGIN FALSE CACHE BOOL
   "Turn off the build of the Detray Vc code" )

# Get it into the current directory.
FetchContent_MakeAvailable( Detray )
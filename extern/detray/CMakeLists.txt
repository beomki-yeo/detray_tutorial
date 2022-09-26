# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

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

set( DETRAY_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the Detray unit tests" )
set( DETRAY_SETUP_DFELIBS FALSE CACHE BOOL
   "Do not set up dfelibs as part of Detray" )
set( DETRAY_SETUP_GOOGLETEST FALSE CACHE BOOL
   "Do not set up GoogleTest as part of Detray" )
set( DETRAY_SETUP_BENCHMARK FALSE CACHE BOOL
   "Do not set up Google Benchmark as part of Detray" )

# Get it into the current directory.
FetchContent_MakeAvailable( Detray )

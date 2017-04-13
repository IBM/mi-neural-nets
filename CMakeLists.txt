# Project name
project(MINeuralNets CXX C)

#  Set current version number.
set (MINeuralNets_VERSION_MAJOR 1)
set (MINeuralNets_VERSION_MINOR 2)
set (MINeuralNets_VERSION_PATCH 1)
set (MINeuralNets_VERSION ${MINeuralNets_VERSION_MAJOR}.${MINeuralNets_VERSION_MINOR}.${MINeuralNets_VERSION_PATCH})

# CMake required version.
cmake_minimum_required(VERSION 3.2)


# =======================================================================
# Set compiler/linker flags.
# =======================================================================
# Add C++11 dependency.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -fopenmp -Wall")

# Check, whether all necessary libraries are linked
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}") 

# =======================================================================
# Find required packages
# =======================================================================
# Add path to cmake dir.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find Boost package
find_package(Boost 1.54 REQUIRED COMPONENTS system thread random  serialization)
# Try to include Boost as system directory to suppress it's warnings
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# Find Eigen package
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

# Find MIC Toolchain
find_package(MIToolchain 1.1 REQUIRED)

# Find MIC Algorithms
find_package(MIAlgorithms 1.1 REQUIRED)

# Find MIC Visualization
find_package(MIVisualization 1.2 REQUIRED)

# =======================================================================
# Find optional packages
# =======================================================================

# Find OpenBLAS
find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
    message(WARNING "-- OpenBLAS not found!")
else(NOT OpenBLAS_FOUND)
	include_directories(${OpenBLAS_INCLUDE_DIR})
	add_definitions(-DOpenBLAS_FOUND=1)
#	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lopenblas") 
#	ADD_DEFINITIONS("-DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK")
endif(NOT OpenBLAS_FOUND)

# Locate GTest
find_package(GTest)
if(NOT GTEST_FOUND)
    message(WARNING "--   GTest not found!")
else(NOT GTEST_FOUND)
	include_directories(${GTEST_INCLUDE_DIRS})
	# This must be set in the root directory for the tests to be run by 'make test' or ctest.
	enable_testing()
endif(NOT GTEST_FOUND)

# Add additional option to cmake.
set(BUILD_UNIT_TESTS ON CACHE BOOL "Build unit tests.")


# =======================================================================
# RPATH settings
# =======================================================================
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# =======================================================================
# Add subdirectories
# =======================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)

add_subdirectory(configs)

#add_subdirectory(datasets)

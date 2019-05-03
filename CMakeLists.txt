cmake_minimum_required(VERSION 2.8)
project(ibmP)

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)
include_directories(${MPI_INCLUDE_PATH})

add_subdirectory(solver)
include_directories(solver)
add_subdirectory(boundary)
include_directories(boundary)
add_subdirectory(exe)

cmake_minimum_required(VERSION 2.8)
project(ibmP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -lfftw3 -lfftw3_threads -lpthread -lm -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -lfftw3 -lfftw3_threads -lpthread -lm ")

add_subdirectory(solver)
include_directories(solver)
add_subdirectory(boundary)
include_directories(boundary)
add_subdirectory(exe)

set(SOURCES
	fluidSolver.cpp mpi_fluidsolver_example.cpp)

add_library(solver ${SOURCES})
target_link_libraries(solver fftw3_mpi fftw3_threads fftw3)



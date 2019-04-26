set(SOURCES
	fluidSolver.cpp)

add_library(solver ${SOURCES})
target_link_libraries(solver fftw3_threads fftw3)



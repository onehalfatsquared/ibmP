set(SOURCES
	ImmersedFiber.cpp)

add_library(boundary ${SOURCES})
target_link_libraries(boundary solver)
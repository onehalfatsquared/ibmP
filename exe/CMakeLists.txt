add_executable(runIBM ibmP.cpp)


target_link_libraries(runIBM boundary solver pthread)

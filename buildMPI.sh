#!/bin/bash
cmake .. \
	-DCMAKE_CXX_COMPILER="mpic++"\
	-DCMAKE_C_COMPILER="mpicc"\
	-DCMAKE_CXX_FLAGS=" -std=c++11 -lfftw3 -lpthread -lm -g"\
	-DCMAKE_C_FLAGS=" -lfftw3 -lpthread -lm "
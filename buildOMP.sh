#!/bin/bash
cmake .. \
	-DCMAKE_CXX_COMPILER="g++"\
	-DCMAKE_C_COMPILER="gcc"\
	-DCMAKE_CXX_FLAGS="-fopenmp -std=c++11 -lfftw3 -lfftw3_threads -lpthread -lm -g"\
	-DCMAKE_C_FLAGS="-fopenmp -lfftw3 -lfftw3_threads -lpthread -lm "

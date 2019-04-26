#pragma once
#include <fftw3.h>
#include <math.h>


//fill in frequency arrays
void fillFreq(int n, int* freq);
//perform a 2d fluid solve. OMP parallel fftw3
void fluidSolve2Domp(int* n, int num_threads);
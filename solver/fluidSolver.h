#pragma once
#include <fftw3.h>
#include <math.h>

//go from (i,j) index to index in a row ordered matrix
int toIndex(int nx, int ny, int i, int j);
//fill in frequency arrays
void fillFreq(int n, int* freq);
//perform a 2d fluid solve. OMP parallel fftw3
void fluidSolve2Domp(int* n, double* d, double mu, int num_threads, double* p, double* u, double*v,
																						double* fu, double* fv);
//perform poisson solve
void poissonSolve(int nx, int ny, double Lx, double Ly, double mu, int* freqX, int* freqY,
							 fftw_complex *fuHat, fftw_complex *fvHat, fftw_complex* pHat, 
							 fftw_complex* uHat, fftw_complex* vHat);
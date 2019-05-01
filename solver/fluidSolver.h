#pragma once
#include <fftw3.h>
#include <math.h>

//go from (i,j) index to index in a row ordered matrix
int toIndex(int nx, int ny, int i, int j);
//fill in frequency arrays
void fillFreq(int n, int* freq);
//fill force arrays with correct indexing
void fillForce3D(int* n, double* fu, double* fv, double* fw, 
							double* fuIn, double* fvIn, double* fwIn);
//perform a 2d fluid solve. OMP parallel fftw3
void fluidSolve2Domp(int* n, double* d, double mu, int num_threads, double* p, double* u, double*v,
																						double* fu, double* fv);
//perform poisson solve 2d 
void poissonSolve2D(int nx, int ny, double Lx, double Ly, double mu, int* freqX, int* freqY,
							 fftw_complex *fuHat, fftw_complex *fvHat, fftw_complex* pHat, 
							 fftw_complex* uHat, fftw_complex* vHat);
//perform poisson solve 3d
void poissonSolve3D(int nx, int ny, int nz, double Lx, double Ly, double Lz, double mu, 
							int* freqX, int* freqY, int* freqZ,
							fftw_complex *fuHat, fftw_complex *fvHat, fftw_complex *fwHat, fftw_complex* pHat, 
							fftw_complex* uHat, fftw_complex* vHat, fftw_complex *wHat);
//perform a 3d fluid solve with omp parallel fftw3
void fluidSolve3Domp(int* n, double* d, double mu, int num_threads, double* p, double* u, double* v, double* w,
																						double* fu, double* fv, double* fw);
//re-arrange indexing to output back to boundary
void makeOutput(int* n, double p0, double* u, double* v, double* w, double* p,
												double* uCalc, double* vCalc, double* wCalc, double* pCalc);
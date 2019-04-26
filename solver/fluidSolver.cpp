#include <fftw3.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>






void fillFreq(int n, int* freq) {
	//fill the frequency vectors
	int m1 = (n-1)/2;
	bool testVal = (n+1)%2;
	int m2 = n/2*(testVal);
	bool count = 0;
	for (int i = 0; i < m1+1; i++) {
		freq[i] = i;
	}
	if (testVal) {
		freq[m1+1] = m2;
		count = 1;
	}
	for (int i = 0; i < m1; i++) {
		freq[m1+1+count+i] = -m1+i;
	}
}



void fluidSolve2Domp(int* n, int num_threads) {
	//solve the fluid equations
	int nx = n[0]; int ny = n[1];

	//create frequency vectors
	int* freqX = new int[nx];
	int* freqY = new int[ny];
	fillFreq(nx, freqX); fillFreq(ny, freqY);

	//declare storage for complex fftw variables
	fftw_complex *pHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *fuHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *fvHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *fwHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *uHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex)); 
	fftw_complex *vHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex)); 
	fftw_complex *wHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));

	//declare storage for real variables - forces/velcoity taken as input?
	double* pOut = new double[nx*ny];

	//create a plan for fftw using supplied number of threads
	fftw_plan p;
	fftw_plan_with_nthreads(num_threads); //if 1, uses serial. every plan after this uses nthreads

	//create plans, do FFT, delete plans. 






















	//free memory
	fftw_free(pHat); fftw_free(fuHat); fftw_free(fvHat); fftw_free(fwHat);
	fftw_free(uHat); fftw_free(vHat); fftw_free(wHat);


	//delete memory
	delete []freqX; delete []freqY; delete []pOut;


}
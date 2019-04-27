#define _USE_MATH_DEFINES
#include <fftw3.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>



int toIndex(int nx, int ny, int i, int j) {
	//returns the row ordered matrix index of element (i,j)
	return nx*i+j;
}


void fillFreq(int n, int* freq) {
	//fill the frequency arrays
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

void poissonSolve(int nx, int ny, double Lx, double Ly, double mu, int* freqX, int* freqY,
							 fftw_complex *fuHat, fftw_complex *fvHat, fftw_complex* pHat, 
							 fftw_complex* uHat, fftw_complex* vHat) {
	//compute pHat, uHat, vHat via poisson solves using FFTed forces
	double kx, ky, k2;
	double fuRe, fuIm, fvRe, fvIm, dRe, dIm;
	int index;
	for (int i = 0; i < nx; i++) {
		kx = freqX[i]*2*M_PI/Lx;
		for (int j = 0; j < ny; j++) {
			ky = freqY[j]*2*M_PI/Ly; 
			k2 = kx*kx+ky*ky;

			//not sure if this is indexed correctly
			index = toIndex(nx,ny,i,j);
			fuRe = fuHat[index][0]; fuIm = fuHat[index][1];
			fvRe = fvHat[index][0]; fvIm = fuHat[index][1];
			if (i == 0 && j == 0) { //0 mode, handle seperately
				dRe = 0.0; dIm = 0.0; k2 = 1.0;
			}
			else{ //compute divfHat
				dRe = -kx*fuIm-ky*fvIm;
        dIm = kx*fuRe+ky*fvRe;
			}
			//solve for pHat
			pHat[index][0] = -dRe/k2; pHat[index][1] = -dIm/k2;
			//solve for uHat
			uHat[index][0] = (fuRe + kx*pHat[index][1])/(mu*k2);
			uHat[index][1] = (fuIm - kx*pHat[index][0])/(mu*k2);
			//solve for vHat
			vHat[index][0] = (fvRe + ky*pHat[index][1])/(mu*k2);
			vHat[index][1] = (fvIm - ky*pHat[index][0])/(mu*k2);
		}
	}

}



void fluidSolve2Domp(int* n, double* d, double mu, int num_threads, double* p, double* u, double*v,
																						double* fu, double* fv) {
	//solve the fluid equations, takes pressure, velocity, forces as input

	//set parameters
	int nx = n[0]; int ny = n[1];
	double dx = d[0]; double dy=d[1];
	double Lx = nx*dx; double Ly = ny*dy;

	//create frequency arrays
	int* freqX = new int[nx];
	int* freqY = new int[ny];
	fillFreq(nx, freqX); fillFreq(ny, freqY);

	//declare storage for complex fftw variables
	fftw_complex *pHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *fuHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *fvHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex));
	fftw_complex *uHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex)); 
	fftw_complex *vHat = (fftw_complex *) fftw_malloc(nx*ny*sizeof(fftw_complex)); 

	//create a plan for fftw using supplied number of threads
	fftw_plan P;
	fftw_plan_with_nthreads(num_threads); //if 1, uses serial. every plan after this uses nthreads

	//fft all of the forces. create, execute, delete plans
	/*Note: using FFTW_ESTIMATE flag potentially gives
	a sub-optimal plan, but it wont overwrite the input array*/
	P = fftw_plan_dft_r2c_2d(nx,ny,fu,fuHat,FFTW_ESTIMATE); 
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_r2c_2d(nx,ny,fv,fvHat,FFTW_ESTIMATE); 
	fftw_execute(P);
	fftw_destroy_plan(P);

	//solve for pHat, vHat, uHat
	poissonSolve(nx, ny, Lx, Ly, mu, freqX, freqY, fuHat, fvHat, pHat, uHat, vHat);

	//fft the pressure and velocity back to real space
	P = fftw_plan_dft_c2r_2d(nx,ny,pHat,p,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_c2r_2d(nx,ny,uHat,u,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_c2r_2d(nx,ny,vHat,v,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	/*Ondre - your example divides by nx*ny*nz. Is this because the IFFT doesn't scale by
	this automatically? You also subtract the p0 from the pressure. Not sure why but 
	I'll do it here */

	double p0 = p[0];
	int N = nx*ny;
	for (int i = 0; i < N; i++) {
		p[i] = (p[i]-p[0])/N;
		u[i] = u[i]/N;
		v[i] = v[i]/N;
	}

	//free memory
	fftw_free(pHat); fftw_free(fuHat); fftw_free(fvHat); 
	fftw_free(uHat); fftw_free(vHat); 


	//delete memory
	delete []freqX; delete []freqY; 

}
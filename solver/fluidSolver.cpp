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

void fillForce3D(int* n, double* fu, double* fv, double* fw, 
							double* fuIn, double* fvIn, double* fwIn) {
	//fill the force arrays to go to the fft

	int nx = n[0]; int ny = n[1]; int nz = n[2];
	int rowIndex, columnIndex;
	for(int i = 0; i < nx; i++) {
		for(int j = 0;j < ny; j++) {
      for(int k = 0;k < nz; k++) {
        rowIndex = k+nz*(j+ny*i);
        columnIndex = i+nx*(j+ny*k);
        fuIn[rowIndex] = fu[columnIndex];
        fvIn[rowIndex] = fv[columnIndex];
        fwIn[rowIndex] = fw[columnIndex];
    	}
   	}
 	}
}

void makeOutput(int* n, double p0, double* u, double* v, double* w, double* p,
												double* uCalc, double* vCalc, double* wCalc, double* pCalc) {
	//re-arrange the indexing to output back to the immersed boundary 
	int nx = n[0]; int ny = n[1]; int nz = n[2];
	double N = nx*ny*nz*1.0;
	int rowIndex; int count = 0;
	double uVal, vVal, wVal, pVal;
	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				rowIndex = k+nz*(j+ny*i);
				uVal = uCalc[rowIndex]/N; u[count] = uVal;
				vVal = vCalc[rowIndex]/N; v[count] = vVal;
				wVal = wCalc[rowIndex]/N; w[count] = wVal;
				pVal = (pCalc[rowIndex]-p0)/N; p[count] = pVal;
				count++;
			}
		}
	}
}

void poissonSolve2D(int nx, int ny, double Lx, double Ly, double mu, int* freqX, int* freqY,
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
	poissonSolve2D(nx, ny, Lx, Ly, mu, freqX, freqY, fuHat, fvHat, pHat, uHat, vHat);

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

void poissonSolve3D(int nx, int ny, int nz, double Lx, double Ly, double Lz, double mu, 
							int* freqX, int* freqY, int* freqZ,
							fftw_complex *fuHat, fftw_complex *fvHat, fftw_complex *fwHat, fftw_complex* pHat, 
							fftw_complex* uHat, fftw_complex* vHat, fftw_complex *wHat) {
	//compute pHat, uHat, vHat, wHat via poisson solves using FFTed forces
	int nzh = nz/2+1;
	double kx, ky, kz, k2;
	double fuRe, fuIm, fvRe, fvIm, fwRe, fwIm, dRe, dIm;
	int rowIndex, columnIndex;
	//#pragma omp parallel for
	for (int i = 0; i < nx; i++) {
		kx = freqX[i]*2.0*M_PI/Lx;
		for (int j = 0; j < ny; j++) {
			ky = freqY[j]*2.0*M_PI/Ly; 
			for (int k = 0; k < nzh; k++) {
				kz = freqZ[k]*2.0*M_PI/Lz;
				k2 = kx*kx+ky*ky+kz*kz;
				rowIndex = k+nzh*(j+ny*i);
        columnIndex = i+nx*(j+ny*k);

				//compute real and imaginary parts of forces
				fuRe = fuHat[rowIndex][0]; fuIm = fuHat[rowIndex][1];
				fvRe = fvHat[rowIndex][0]; fvIm = fvHat[rowIndex][1];
				fwRe = fwHat[rowIndex][0]; fwIm = fwHat[rowIndex][1];

				if (i == 0 && j == 0 && k == 0) { //0 mode, handle seperately
					dRe = 0.0; dIm = 0.0; k2 = 1.0;
				}
				else{ //compute divfHat
					dRe = -kx*fuIm-ky*fvIm-kz*fwIm;
	        dIm = kx*fuRe+ky*fvRe+kz*fwRe;
				}
				//solve for pHat
				pHat[rowIndex][0] = -dRe/k2; pHat[rowIndex][1] = -dIm/k2;
				//solve for uHat
				uHat[rowIndex][0] = (fuRe + kx*pHat[rowIndex][1])/(mu*k2);
				uHat[rowIndex][1] = (fuIm - kx*pHat[rowIndex][0])/(mu*k2);
				//solve for vHat
				vHat[rowIndex][0] = (fvRe + ky*pHat[rowIndex][1])/(mu*k2);
				vHat[rowIndex][1] = (fvIm - ky*pHat[rowIndex][0])/(mu*k2);
				//solve for wHat
				wHat[rowIndex][0] = (fwRe + kz*pHat[rowIndex][1])/(mu*k2);
				wHat[rowIndex][1] = (fwIm - kz*pHat[rowIndex][0])/(mu*k2);
			}
		}
	}

}

void fluidSolve3Domp(int* n, double* d, double mu, int num_threads, double* p, double* u, double* v, double* w,
																						double* fu, double* fv, double* fw) {
	//solve the fluid equations, takes pressure, velocity, forces as input

	//set parameters
	int nx = n[0]; int ny = n[1]; int nz = n[2];
	double dx = d[0]; double dy=d[1]; double dz = d[2];
	double Lx = nx*dx; double Ly = ny*dy; double Lz = nz*dz;
	int nzh = nz/2 + 1; //half of the z values b/c conjugate symmetry

	//create frequency arrays
	int* freqX = new int[nx];
	int* freqY = new int[ny];
	int* freqZ = new int[nz];
	fillFreq(nx, freqX); fillFreq(ny, freqY); fillFreq(nz, freqZ);

	//create force arrays indexed in the correct way to pass to poisson solve
	double* fuIn = new double[nx*ny*nz];
	double* fvIn = new double[nx*ny*nz];
	double* fwIn = new double[nx*ny*nz];

	//create velocity arrays for calculations
	double* uCalc = new double[nx*ny*nz];
	double* vCalc = new double[nx*ny*nz];
	double* wCalc = new double[nx*ny*nz];
	double* pCalc = new double[nx*ny*nz];

	//fill force arrays
	fillForce3D(n, fu, fv, fw, fuIn, fvIn, fwIn);

	//declare storage for complex fftw variables
	fftw_complex *pHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex));
	memset(pHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *fuHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex));
	memset(fuHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *fvHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex));
	memset(fvHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *fwHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex));
	memset(fwHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *uHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex)); 
	memset(uHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *vHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex)); 
	memset(vHat,0,sizeof(fftw_complex)*nx*ny*nzh);
	fftw_complex *wHat = (fftw_complex *) fftw_malloc(nx*ny*nzh*sizeof(fftw_complex)); 
	memset(wHat,0,sizeof(fftw_complex)*nx*ny*nzh);

	//create a plan for fftw using supplied number of threads
	fftw_plan P;
	fftw_plan_with_nthreads(num_threads); //if 1, uses serial. every plan after this uses nthreads

	//fft all of the forces. create, execute, delete plans
	/*Note: using FFTW_ESTIMATE flag potentially gives
	a sub-optimal plan, but it wont overwrite the input array*/
	P = fftw_plan_dft_r2c_3d(nx,ny,nz,fuIn,fuHat,FFTW_ESTIMATE); 
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_r2c_3d(nx,ny,nz,fvIn,fvHat,FFTW_ESTIMATE); 
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_r2c_3d(nx,ny,nz,fwIn,fwHat,FFTW_ESTIMATE); 
	fftw_execute(P);
	fftw_destroy_plan(P);

	//solve for pHat, vHat, uHat, wHat
	poissonSolve3D(nx, ny, nz, Lx, Ly, Lz, mu, freqX, freqY, freqZ,
	 fuHat, fvHat, fwHat, pHat, uHat, vHat, wHat);

	//fft the pressure and velocity back to real space
	P = fftw_plan_dft_c2r_3d(nx,ny,nz,pHat,pCalc,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_c2r_3d(nx,ny,nz,uHat,uCalc,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_c2r_3d(nx,ny,nz,vHat,vCalc,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	P = fftw_plan_dft_c2r_3d(nx,ny,nz,wHat,wCalc,FFTW_ESTIMATE);
	fftw_execute(P);
	fftw_destroy_plan(P);

	//re-arrange the indexing of velocities for output
	double p0 = pCalc[0];
	makeOutput(n, p0, u, v, w, p, uCalc, vCalc, wCalc, pCalc);

	//free memory
	fftw_free(pHat); fftw_free(fuHat); fftw_free(fvHat); fftw_free(fwHat);
	fftw_free(uHat); fftw_free(vHat); fftw_free(wHat);


	//delete memory
	delete []freqX; delete []freqY; delete []freqZ;
	delete []fuIn; delete []fvIn; delete []fwIn;
	delete []uCalc; delete []vCalc; delete []wCalc; delete []pCalc;

}
/* Parallel Implmentation of the immersed boundary method */
#include <fftw3.h>
#include <math.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "fluidSolver.h"
#include "ImmersedFiber.h"



int main(int argc, char* argv[]) {

	//check for input //todo - what is input?
	if (argc != 2) {
		fprintf(stderr, "Usage: <?> %s\n", argv[0]);
		return 1;
	}

	// Input parameters. Will make into an input file later. 
    double sphereRadius = 1.0;
    double springStiff = 1.0;
    double ae = -1.0;
    double be = 1.0;
    int n = 32;
    double mu=1.0;
    double dt=1e-4;
    double until = 1.0;
    double h = (be-ae)/((double)n);

    //Initialize Eulerian grid
    double *xEpts = (double *) malloc(n*sizeof(double));
    double *yEpts = (double *) malloc(n*sizeof(double));
    double *zEpts = (double *) malloc(n*sizeof(double));
    for (int iPt=0; iPt < n; iPt++){
			xEpts[iPt]=ae+h*iPt;
			yEpts[iPt]=ae+h*iPt;
			zEpts[iPt]=ae+h*iPt;
    }

    // Initialize arrays for the spread forces
    // The stacking of the arrays is fluidIndex = (n*n)*(zIndex)+n*(yIndex)+xIndex
    double *gridFX = (double *) malloc(n*n*n*sizeof(double));
    double *gridFY = (double *) malloc(n*n*n*sizeof(double));
    double *gridFZ = (double *) malloc(n*n*n*sizeof(double));
    int NIB = (int)floor(4*M_PI*sphereRadius/h); // two points per meshwidth
    ImmersedFiber Fib = ImmersedFiber(sphereRadius, springStiff, 1.5, NIB);
    Fib.calcForces();
    Fib.spreadForces(gridFX, gridFY, gridFZ, xEpts, n, yEpts, n, zEpts, n);
    //outputs forces in x and y
    /*
    std::ofstream fout1("xGridForce.txt");
    std::ofstream fout2("yGridForce.txt");
    for (int iPt=0; iPt < n*n*n; iPt++){
			fout1 << gridFX[iPt] << std::endl;
			fout2 << gridFY[iPt] << std::endl;
    } 
    */

	//initialize parallel fft library
	int fftw_init_threads(void);

	//call solver routine //
	int N[3]; N[0]=N[1]=N[2] = n;
	double H[3]; H[0]=H[1]=H[2] = h;
	int num_threads = 4;
	double *p = (double *) malloc(n*n*n*sizeof(double));
	double *u = (double *) malloc(n*n*n*sizeof(double));
  double *v = (double *) malloc(n*n*n*sizeof(double));
  double *w = (double *) malloc(n*n*n*sizeof(double));
	fluidSolve3Domp(N, H, mu,num_threads, p, u, v, w, gridFX, gridFY, gridFZ);
	//velocity is now in u,v,w



	//free the memory
	free(gridFX); free(gridFY); free(gridFZ); free(u); free(v); free(w); free(p);
	free(xEpts); free(yEpts); free(zEpts);


	return 0;
}
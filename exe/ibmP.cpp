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
#include <vector> 
#include <omp.h>


int main(int argc, char* argv[]) {
    // Input parameters. Will make into an input file later. 
    std::ifstream infile;
    infile.open("inputs.txt");
    std::string name;
    double var1;
    std::vector <double> values;
    while (infile >> name >> var1)
    {
        values.push_back(var1);
    }
    double L = values[0];
    double kel = values[1];
    double kbend = values[2];
    double ae = values[3];
    double be = values[4];
    int n = (int) (values[5]+1e-5);
    double mu = values[6];
    double dt = values[7];
    double until = values[8];
    int random = (int) values[9];
    //printf("%f %f %f %f %f %d %f % f %f",L, kel, kbend, ae, be, n, mu, dt, until);
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
    int NIB = (int)floor(2.0*L/h); // two points per meshwidth
    ImmersedFiber Fib = ImmersedFiber(L, kel, kbend, NIB, random);
    double start = omp_get_wtime();
    Fib.calcForces(random);
    double forTime = omp_get_wtime();
    printf("Calc force time is %f \n", forTime-start);
    Fib.spreadForces(gridFX, gridFY, gridFZ, xEpts, n, yEpts, n, zEpts, n);
    double sTime = omp_get_wtime();
    printf("Spread time is %f \n", sTime-forTime);
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
        double fTime = omp_get_wtime();
    	printf("Fluid solve time is %f \n", fTime-sTime);
	//velocity is now in u,v,w
        // Interpolate velocity and get update points
	Fib.getBoundaryVelocity(u, v, w, xEpts, n, yEpts, n, zEpts, n);
	Fib.updatePoints(dt);
	double iTime = omp_get_wtime();

	//free the memory
	free(gridFX); free(gridFY); free(gridFZ); free(u); free(v); free(w); free(p);
	free(xEpts); free(yEpts); free(zEpts);
	
	printf("Interpolate time is %f \n", iTime-fTime);

	return 0;
}

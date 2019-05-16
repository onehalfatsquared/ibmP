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

void readInput(std::string file, double& L, double& kel, double& kbend, double& ae, double& be, int& n, int &Nfib, double& mu, double& dt, double& until, int& random, double& h) {
	//read the parameters from file
	std::ifstream infile;
  infile.open(file);
  std::string name;
  double var1;
  std::vector <double> values;
  while (infile >> name >> var1)
  {
    values.push_back(var1);
  }
  L = values[0];
  kel = values[1];
  kbend = values[2];
  ae = values[3];
  be = values[4];
  n = (int) (values[5]+1e-5);
  Nfib = (int) (values[6]+1e-5);
  mu = values[7];
  dt = values[8];
  until = values[9];
  random = (int) values[10];
  //printf("%f %f %f %f %f %d %f % f %f",L, kel, kbend, ae, be, n, mu, dt, until);
  h = (be-ae)/((double)n);
}

int main(int argc, char* argv[]) {
  // Get input parameters from specified file
	if (argc != 2) {
		fprintf(stderr, "Usage: <Input File> %s\n", argv[0]);
		return 1;
	}
	std::string file = (argv[1]);
	double L, kel, kbend, ae, be, mu, dt, until, h;
	int n, Nfib, random;
	readInput(file, L, kel, kbend, ae, be, n, Nfib, mu, dt, until,random, h);

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

  //initialize the immersed fiber
  int NIB = (int)floor(2.0*L/h); // two points per meshwidth
  if (random){ NIB=1000000/Nfib;}
  ImmersedFiber Fib = ImmersedFiber(L, kel, kbend, NIB, Nfib, random);

  //time force calculation
  double start = omp_get_wtime();
  Fib.calcForces(random);
  double forTime = omp_get_wtime();
  printf("Calc force time is %f\n", forTime-start);

  //time force spreading
  Fib.spreadForces(gridFX, gridFY, gridFZ, xEpts, n, yEpts, n, zEpts, n);
  double sTime = omp_get_wtime();
  printf("Spread time is %f\n", sTime-forTime);
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

	//initialize arrays for solver routine
	int N[3]; N[0]=N[1]=N[2] = n;
	double H[3]; H[0]=H[1]=H[2] = h;
	int num_threads;
	num_threads = omp_get_max_threads();
	double *p = (double *) malloc(n*n*n*sizeof(double));
	double *u = (double *) malloc(n*n*n*sizeof(double));
  	double *v = (double *) malloc(n*n*n*sizeof(double));
  	double *w = (double *) malloc(n*n*n*sizeof(double));

  //call and time fluid solve
	fluidSolve3Domp(N, H, mu,num_threads, p, u, v, w, gridFX, gridFY, gridFZ);
  double fTime = omp_get_wtime();
  printf("Fluid solve time is %f \n", fTime-sTime);
	//velocity is now in u,v,w

  // time interpolation of velocity and update points
	Fib.getBoundaryVelocity(u, v, w, xEpts, n, yEpts, n, zEpts, n);
	Fib.updatePoints(dt);
	double iTime = omp_get_wtime();
	printf("Interpolate time is %f \n", iTime-fTime);

	//free the memory and return
	free(gridFX); free(gridFY); free(gridFZ); free(u); free(v); free(w); free(p);
	free(xEpts); free(yEpts); free(zEpts);
	
	return 0;
}

/* Parallel Implmentation of the immersed boundary method */
#include <fftw3.h>
#include <math.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "fluidSolver.h"









int main(int argc, char* argv[]) {

	//check for input //todo - what is input?
	if (argc != 2) {
		fprintf(stderr, "Usage: <?> %s\n", argv[0]);
		return 1;
	}

	//set parameters //todo - write routines to initialize


	//initialize parallel fft library
	int fftw_init_threads(void);

	//call solver routine //todo - write solvers


	//debug fluid solver routines here
	int n[2]; n[0]=n[1]=128;
	double d[2]; d[0]=d[2]=1/128;
	double mu = 0.05;
	double* p = new double[n[0]*n[1]*sizeof(double)];
	double* u = new double[n[0]*n[1]*sizeof(double)];
	double* v = new double[n[0]*n[1]*sizeof(double)];
	double* fu = new double[n[0]*n[1]*sizeof(double)];
	double* fv = new double[n[0]*n[1]*sizeof(double)];
	fluidSolve2Domp(n, d, mu, 2, p, u, v, fu, fv);
	delete []p;delete []u;delete []v;delete []fu;delete []fv;






















	return 0;
}
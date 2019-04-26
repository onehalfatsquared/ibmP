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
	int n[2]; n[0]=n[1]=6;
	fluidSolve2Domp(n, 2);






















	return 0;
}
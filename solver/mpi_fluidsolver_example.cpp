#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <fftw3.h>

#define NUM_PLANS2 5
#define NUM_PLANS3 7

int threads_ok;
//int main(int argc, char **argv){
//  MPI_Init(&argc, &argv);
//  fftw_complex a[2];
//
//  int fftw_mpi_init(void);
//
//  MPI_Finalize();
//}



void init_data2D(double** physical, fftw_complex** hatted, const ptrdiff_t* ns,
    ptrdiff_t* local_n, fftw_plan* P){
  // We wish intialize the variables U = (u,v), p, F = (fu, fv)
  ptrdiff_t alloc_local, alloc_local_tr;


  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_2d(ns[0], ns[1]/2+1, MPI_COMM_WORLD, local_n,
      local_n+1);
  alloc_local_tr = fftw_mpi_local_size_2d(ns[1]/2+1, ns[0], MPI_COMM_WORLD, local_n+2,
      local_n+3);
 
  for(int i=0; i<5; i++){
    physical[i] = fftw_alloc_real(2*alloc_local);
    hatted[i] = fftw_alloc_complex(alloc_local_tr);
  }

  /* create plan for out-of-place r2c DFT */
  P[0] = fftw_mpi_plan_dft_r2c_2d(ns[0], ns[1], physical[3], hatted[3], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT); // fx
  P[1] = fftw_mpi_plan_dft_r2c_2d(ns[0], ns[1], physical[4], hatted[4], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT); // fy

  P[2] = fftw_mpi_plan_dft_c2r_2d(ns[0], ns[1], hatted[0], physical[0], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //ux
  P[3] = fftw_mpi_plan_dft_c2r_2d(ns[0], ns[1], hatted[1], physical[1], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //uy
  P[4] = fftw_mpi_plan_dft_c2r_2d(ns[0], ns[1], hatted[2], physical[2], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //p
}

void clean_data2D(double** physical, fftw_complex** hatted, fftw_plan* P){
  // We wish free the variables U = (u,v), p, F = (fu, fv)
  for(int i=0; i<5; i++){
    fftw_free(physical[i]);
    fftw_free(hatted[i]);
  }
  
  for(int i=0; i<NUM_PLANS2; i++) fftw_destroy_plan(P[i]);
}


inline ptrdiff_t freq(ptrdiff_t i, ptrdiff_t nh){
  return (nh-abs(i-nh))*(2*(i<=nh)-1);
}

void poissonSolve2D_mpi(const ptrdiff_t* ns, const double* Ls, double mu,  fftw_complex** hatted, ptrdiff_t* local_n){
  //compute pHat, uHat, vHat via poisson solves using FFTed forces
  double kx, ky, k2;
  double fuRe, fuIm, fvRe, fvIm, dRe, dIm;
  int index;

  #pragma omp parallel for
  for (int i = 0; i < local_n[2]; i++) {
    kx = freq(i+local_n[3], ns[1]/2)*2*M_PI/Ls[1];
    for (int j = 0; j < ns[0]/2+1; j++) {
      ky = j*2*M_PI/Ls[0];
      k2 = kx*kx+ky*ky;

      //not sure if this is indexed correctly
      index = i*ns[0] +j;
      fuRe = hatted[3][index][0]; fuIm = hatted[3][index][1];
      fvRe = hatted[4][index][0]; fvIm = hatted[4][index][1];
      if (i == 0 && j == 0) { //0 mode, handle seperately
        dRe = 0.0; dIm = 0.0; k2 = 1.0;
      }
      else{ //compute divfHat
        dRe = -kx*fuIm-ky*fvIm;
        dIm = kx*fuRe+ky*fvRe;
      }
      k2 *= ns[0]*ns[1];
      //solve for pHat
      hatted[2][index][0] = -dRe/k2; hatted[2][index][1] = -dIm/k2;
      //solve for uHat
      hatted[0][index][0] = (fuRe + kx*hatted[2][index][1])/(mu*k2);
      hatted[0][index][1] = (fuIm - kx*hatted[2][index][0])/(mu*k2);
      //solve for vHat
      hatted[1][index][0] = (fvRe + ky*hatted[2][index][1])/(mu*k2);
      hatted[1][index][1] = (fvIm - ky*hatted[2][index][0])/(mu*k2);
    }

  }
}

void fluid_solve_2D_mpi(double** physical, fftw_complex** hatted, const ptrdiff_t* ns, double*
    Ls, double mu, ptrdiff_t* local_n, fftw_plan* P){
  
  for(int i=0; i<2; i++) fftw_execute(P[i]);

  poissonSolve2D_mpi(ns, Ls, mu, hatted, local_n);

  for(int i=2; i<NUM_PLANS2; i++) fftw_execute(P[i]);
}

void init_data3D(double** physical, fftw_complex** hatted, const ptrdiff_t* ns,
    ptrdiff_t* local_n, fftw_plan* P){
  // We wish intialize the variables U = (u,v, w), p, F = (fu, fv, fw)
  ptrdiff_t alloc_local, alloc_local_tr;


  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d(ns[0], ns[1], ns[2]/2+1, MPI_COMM_WORLD, local_n,
      local_n+1);
  alloc_local_tr = fftw_mpi_local_size_3d(ns[0], ns[2]/2+1, ns[1], MPI_COMM_WORLD, local_n+2,
      local_n+3);
  
  for(int i=0; i<7; i++){
    physical[i] = fftw_alloc_real(2*alloc_local);
    hatted[i] = fftw_alloc_complex(alloc_local_tr);
  }

  /* create plan for out-of-place r2c DFT */
  P[0] = fftw_mpi_plan_dft_r2c_3d(ns[0], ns[1], ns[2], physical[4], hatted[4], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT); // fx
  P[1] = fftw_mpi_plan_dft_r2c_3d(ns[0], ns[1], ns[2], physical[5], hatted[5], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT); // fy
  P[2] = fftw_mpi_plan_dft_r2c_3d(ns[0], ns[1], ns[2], physical[6], hatted[6], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT); // fy

  P[3] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[2], ns[1], hatted[0], physical[0], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //ux
  P[4] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[2], ns[1], hatted[1], physical[1], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //uy
  P[5] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[2], ns[1], hatted[2], physical[2], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //p
  P[6] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[2], ns[1], hatted[3], physical[3], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //p
}

void clean_data3D(double** physical, fftw_complex** hatted, fftw_plan* P){
  // We wish free the variables U = (u,v), p, F = (fu, fv)
  
  for(int i=0; i<7; i++){
    fftw_free(physical[i]);
    fftw_free(hatted[i]);
  }
  
  for(int i=0; i<NUM_PLANS3; i++) fftw_destroy_plan(P[i]);
}


void poissonSolve3D_mpi(const ptrdiff_t* ns, const double* Ls, double mu,  fftw_complex** hatted, ptrdiff_t* local_n){
  //compute pHat, uHat, vHat, wHat via poisson solves using FFTed forces
  double kx, ky, kz, k2;
  double fuRe, fuIm, fvRe, fvIm, fwRe, fwIm, dRe, dIm;
  int index;

#pragma omp parallel for
  for (int i = 0; i < local_n[2]; i++) {
    kx = freq(i+local_n[3], ns[0]/2)*2*M_PI/Ls[0];
    for (int j = 0; j < ns[2]; j++) {
      ky = freq(j,ns[1]/2)*2*M_PI/Ls[2];
      for (int k = 0; k < ns[1]/2+1; k++) {
        kz = k*2*M_PI/Ls[1];
        k2 = kx*kx+ky*ky+kz*kz;

        //not sure if this is indexed correctly
        index = (i*ns[2] +j)*ns[1] + k;
        fuRe = hatted[4][index][0]; fuIm = hatted[4][index][1];
        fvRe = hatted[5][index][0]; fvIm = hatted[5][index][1];
        fwRe = hatted[6][index][0]; fwIm = hatted[6][index][1];
        if (i == 0 && j == 0 && k == 0) { //0 mode, handle seperately
          dRe = 0.0; dIm = 0.0; k2 = 1.0;
        }
        else{ //compute divfHat
          dRe = -kx*fuIm-ky*fvIm-kz*fwIm;
          dIm = kx*fuRe+ky*fvRe+kz*fwRe;
        }
        k2 *= ns[0]*ns[1]*ns[2];
        //solve for pHat
        hatted[3][index][0] = -dRe/k2; hatted[3][index][1] = -dIm/k2;
        //solve for uHat
        hatted[0][index][0] = (fuRe + kx*hatted[3][index][1])/(mu*k2);
        hatted[0][index][1] = (fuIm - kx*hatted[3][index][0])/(mu*k2);
        //solve for vHat
        hatted[1][index][0] = (fvRe + ky*hatted[3][index][1])/(mu*k2);
        hatted[1][index][1] = (fvIm - ky*hatted[3][index][0])/(mu*k2);
        //solve for wHat
        hatted[2][index][0] = (fvRe + kz*hatted[3][index][1])/(mu*k2);
        hatted[2][index][1] = (fvIm - kz*hatted[3][index][0])/(mu*k2);
      }
    }

  }
}


void fluid_solve_3D_mpi(double** physical, fftw_complex** hatted, const ptrdiff_t* ns, const double*
    Ls, double mu, ptrdiff_t* local_n, fftw_plan* P){
  
  for(int i=0; i<3; i++) fftw_execute(P[i]);

  poissonSolve3D_mpi(ns, Ls, mu, hatted, local_n);

  for(int i=3; i<NUM_PLANS3; i++) fftw_execute(P[i]);
}

int main(int argc, char **argv){
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  threads_ok = provided >=MPI_THREAD_FUNNELED;
  if (threads_ok) threads_ok = fftw_init_threads();
  int fftw_mpi_init(void);

  if (threads_ok) fftw_plan_with_nthreads(omp_get_max_threads());

  const ptrdiff_t ns[2] = {4, 8};
  const double Ls[2] = {1, 1}, mu = 1.;
  fftw_plan P[NUM_PLANS2];

  double *physical[5];
  fftw_complex  *hatted[5];
  
  ptrdiff_t local_n[4];
  init_data2D(physical, hatted, ns, local_n, P);
  //printf("Hello %d %d %d %d\n", local_n[0], local_n[1], local_n[2], local_n[3]);
  /*initialize u to */
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      physical[3][i*2*(ns[1]/2+1)+j] =1.;

      physical[4][i*2*(ns[1]/2+1)+j] = sin(2*M_PI*(i+local_n[1])/ns[0]);
      physical[0][i*2*(ns[1]/2+1)+j] = 2*M_PI*cos(2*M_PI*(i+local_n[1])/ns[0]);
      //printf("%f \t", physical[3][i*2*(ns[1]/2+1)+j]);
      printf("%f \t", physical[4][i*2*(ns[1]/2+1)+j]);
    }
    printf("\n");
  }
  
  fftw_execute(P[0]);
  fftw_execute(P[1]);
  double kx, ky, k2;
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[2]; i++){
    kx = freq(i+local_n[3], ns[1]/2)*2*M_PI/Ls[1];
    for (ptrdiff_t j = 0; j <ns[0]; j++){
      ky = j*2*M_PI/Ls[0];
      hatted[1][i*ns[0]+j][0] = hatted[3][i*ns[0]+j][0]/ns[0]/ns[1];
      hatted[1][i*ns[0]+j][1] = hatted[3][i*ns[0]+j][1]/ns[0]/ns[1];
      //printf("%f %f\t", hatted[1][i*ns[0]+j][0], hatted[1][i*ns[0]+j][1]); 


      hatted[2][i*ns[0]+j][0] = ky*hatted[4][i*ns[0]+j][1]/ns[0]/ns[1];
      hatted[2][i*ns[0]+j][1] = -ky*hatted[4][i*ns[0]+j][0]/ns[0]/ns[1];
      printf("%f %f\t", hatted[2][i*ns[0]+j][0], hatted[2][i*ns[0]+j][1]); 
    }
    printf("\n");
  }

  fftw_execute(P[3]);
  fftw_execute(P[4]);

  double error =0.;

#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      //error += (physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j])*(physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j]) ;
      printf("%f \t", physical[2][i*2*(ns[1]/2+1)+j]);
      error += (physical[0][i*2*(ns[1]/2+1)+j] -physical[2][i*2*(ns[1]/2+1)+j])*(physical[0][i*2*(ns[1]/2+1)+j] -physical[2][i*2*(ns[1]/2+1)+j]) ;
    }
    printf("\n");
  }
  //printf("Hello \n");
  MPI_Barrier(MPI_COMM_WORLD);
  clean_data2D(physical, hatted, P);
  printf(" Error is %f \n", error);
  MPI_Finalize();
}

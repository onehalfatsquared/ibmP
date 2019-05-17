// mpic++ -fopenmp -std=c++11 -I$HOME/Packages/include/ mpi_fluidsolver_example.cpp  -L$HOME/Packages/lib -lfftw3_mpi -lfftw3 -lfftw3_omp -lm
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


void print_array3D(double *a, int nx, int ny, int nz){
  for(int i=0; i<nx; i++){
    for(int j=0; j<ny; j++){
      for(int k=0; k<nz; k++){
        printf("%f\t", a[(i*ny+j)*2*(nz/2+1) + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("--------------------------------------\n");
}

void print_hat_array3D(fftw_complex *a, const ptrdiff_t *ns){
  for(int i=0; i<ns[1]; i++){
    for(int j=0; j<ns[0]; j++){
      for(int k=0; k<ns[2]/2+1; k++){
        printf("%5.2f+i %5.2f\t", a[(i*ns[0]+j)*(ns[2]/2+1)+k][0], a[(i*ns[0]+j)*(ns[2]/2+1)+k][1]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("--------------------------------------\n");
}
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
    for (int j = 0; j < ns[0]; j++) {
      ky = freq(j, ns[0]/2)*2*M_PI/Ls[0];
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

  P[3] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[1], ns[2], hatted[0], physical[0], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //ux
  P[4] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[1], ns[2], hatted[1], physical[1], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //uy
  P[5] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[1], ns[2], hatted[2], physical[2], MPI_COMM_WORLD,
      FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN); //p
  P[6] = fftw_mpi_plan_dft_c2r_3d(ns[0], ns[1], ns[2], hatted[3], physical[3], MPI_COMM_WORLD,
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
  for (ptrdiff_t i =0; i < local_n[2]; i++){
    kx = freq(i+local_n[3], ns[1]/2)*2.*M_PI/Ls[1];
    for (ptrdiff_t j = 0; j <ns[0]; j++){
      ky = freq(j, ns[0]/2)*2.*M_PI/Ls[0];
      for (ptrdiff_t k = 0; k <ns[2]/2+1; k++){
        kz = freq(k, ns[2]/2)*2.*M_PI/Ls[2];
        k2 = kx*kx+ky*ky+kz*kz;
        
        printf("kx = %f, ky = %f, kz = %f, k^2 = %f,\n", kx,ky,kz,k2);
        //not sure if this is indexed correctly
        index = (i*ns[0]+j)*(ns[2]/2+1)+k;
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
        hatted[2][index][0] = (fwRe + kz*hatted[3][index][1])/(mu*k2);
        hatted[2][index][1] = (fwIm - kz*hatted[3][index][0])/(mu*k2);
      }
    }

  }
}


void fluid_solve_3D_mpi(double** physical, fftw_complex** hatted, const ptrdiff_t* ns, const double*
    Ls, double mu, ptrdiff_t* local_n, fftw_plan* P){
  
  for(int i=0; i<3; i++) fftw_execute(P[i]);
  //print_hat_array3D(hatted[4], ns);
  //print_hat_array3D(hatted[5], ns);
  //print_hat_array3D(hatted[6], ns);
  poissonSolve3D_mpi(ns, Ls, mu, hatted, local_n);

  for(int i=3; i<NUM_PLANS3; i++) fftw_execute(P[i]);
}

void twoD_Example(){
  const ptrdiff_t ns[2] = {2048,2048};
  const double Ls[2] = {1, 1}, mu = 1.;
  fftw_plan P[NUM_PLANS2];

  double *physical[5];
  fftw_complex  *hatted[5];
  
  ptrdiff_t local_n[4];
  init_data2D(physical, hatted, ns, local_n, P);
  double tt = MPI_Wtime();
  //printf("Hello %d %d %d %d\n", local_n[0], local_n[1], local_n[2], local_n[3]);
  /*initialize u to */
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      physical[3][i*2*(ns[1]/2+1)+j] =1.;

      physical[4][i*2*(ns[1]/2+1)+j] = sin(2*M_PI*(i+local_n[1])/ns[0]);
      physical[0][i*2*(ns[1]/2+1)+j] = 2*M_PI*cos(2*M_PI*(i+local_n[1])/ns[0])/Ls[0];
      //physical[4][i*2*(ns[1]/2+1)+j] = sin(2*M_PI*(j)/ns[1]);
      //physical[0][i*2*(ns[1]/2+1)+j] = 2*M_PI*cos(2*M_PI*(j)/ns[1])/Ls[1];
      //printf("%f \t", physical[3][i*2*(ns[1]/2+1)+j]);
      //printf("%f \t", physical[0][i*2*(ns[1]/2+1)+j]);
    }
    //printf("\n");
  }
  
   // printf("\n");
  fftw_execute(P[0]);
  fftw_execute(P[1]);
  double kx, ky, k2;
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[2]; i++){
    kx = freq(i+local_n[3], ns[1]/2)*2*M_PI/Ls[1];
    //printf("i = %d, kx = %f, n1/2 = %d\n", i, kx, ns[1]/2);
    for (ptrdiff_t j = 0; j <ns[0]; j++){
      ky = freq(j, ns[0]/2)*2*M_PI/Ls[0];
      hatted[1][i*ns[0]+j][0] = hatted[3][i*ns[0]+j][0]/ns[0]/ns[1];
      hatted[1][i*ns[0]+j][1] = hatted[3][i*ns[0]+j][1]/ns[0]/ns[1];
      //printf("%f %f\t", hatted[1][i*ns[0]+j][0], hatted[1][i*ns[0]+j][1]); 


      hatted[2][i*ns[0]+j][0] = -ky*hatted[4][i*ns[0]+j][1]/ns[0]/ns[1];
      hatted[2][i*ns[0]+j][1] = ky*hatted[4][i*ns[0]+j][0]/ns[0]/ns[1];
      //printf("%f %f\t", hatted[2][i*ns[0]+j][0], hatted[2][i*ns[0]+j][1]); 
    }
    //printf("\n");
  }

   // printf("\n");
  fftw_execute(P[3]);
  fftw_execute(P[4]);

  double error =0.;

#pragma omp parallel for reduction(+:error)
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      //error += (physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j])*(physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j]) ;
      //printf("%f \t", physical[2][i*2*(ns[1]/2+1)+j]);
      error += (physical[0][i*2*(ns[1]/2+1)+j] -physical[2][i*2*(ns[1]/2+1)+j])*(physical[0][i*2*(ns[1]/2+1)+j] -physical[2][i*2*(ns[1]/2+1)+j]) ;
    }
    //printf("\n");
  }

  //printf("Hello \n");
  MPI_Barrier(MPI_COMM_WORLD);
  
  printf("Time elapsed %f\n", MPI_Wtime()-tt);
  clean_data2D(physical, hatted, P);
  printf(" Error is %f \n", error);
}

void threeD_Example(){
  int n = 8;
  const ptrdiff_t ns[3] = {n,n,n};
  const double Ls[3] = {1, 1, 1}, mu = 1.;
  ptrdiff_t real_idx, f_idx;
  fftw_plan P[NUM_PLANS3];

  double *physical[7];
  fftw_complex  *hatted[7];
  
  ptrdiff_t local_n[4];
  init_data3D(physical, hatted, ns, local_n, P);
  double tt = MPI_Wtime();
  //printf("Hello %d %d %d %d\n", local_n[0], local_n[1], local_n[2], local_n[3]);
  /*initialize u to */
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      for (ptrdiff_t k = 0; k <ns[2]; k++){
        real_idx = (i*ns[1]+j)*2*(ns[2]/2+1) + k;
        physical[4][real_idx] =1.;

        physical[5][real_idx] = sin(2*M_PI*(i+local_n[1])/ns[0]);
        physical[0][real_idx] = 2*M_PI*cos(2*M_PI*(i+local_n[1])/ns[0]);
        //physical[5][real_idx] = pow(sin(2*M_PI*(i+local_n[1])/ns[0]),5);
        //physical[0][real_idx] = 10*M_PI*pow(sin(2*M_PI*(i+local_n[1])/ns[0]),4)*cos(2*M_PI*(i+local_n[1])/ns[0]);
        //printf("%f \t", physical[3][i*2*(ns[1]/2+1)+j]);
        //printf("%f \t", physical[0][real_idx]);
      }
    //printf("\n");
    }
    //printf("\n");
    //printf("\n");
  }
  
  fftw_execute(P[0]);
  fftw_execute(P[1]);
  //print_array3D(physical[5],ns[0],ns[1],ns[2]);
  //print_hat_array3D(hatted[5],ns);
  double kx, ky, kz, k2;
#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[2]; i++){
    kx = freq(i+local_n[3], ns[1]/2)*2*M_PI/Ls[1];
    for (ptrdiff_t j = 0; j <ns[0]; j++){
      ky = freq(j, ns[0]/2)*2*M_PI/Ls[0];
      for (ptrdiff_t k = 0; k <ns[2]/2+1; k++){
        kz = freq(k, ns[2]/2)*2*M_PI/Ls[2];
        f_idx = (i*ns[0]+j)*(ns[2]/2+1)+k;
        hatted[1][f_idx][0] = hatted[4][f_idx][0]/ns[0]/ns[1]/ns[2];
        hatted[1][f_idx][1] = hatted[4][f_idx][1]/ns[0]/ns[1]/ns[2];
        //printf("%f %f\t", hatted[1][i*ns[0]+j][0], hatted[1][i*ns[0]+j][1]); 


        hatted[2][f_idx][0] = -ky*hatted[5][f_idx][1]/ns[0]/ns[1]/ns[2];
        hatted[2][f_idx][1] = ky*hatted[5][f_idx][0]/ns[0]/ns[1]/ns[2];
        //printf("%f %f\t", hatted[2][f_idx][0], hatted[2][f_idx][1]); 
      }
      //printf("\n");
    }
    //printf("\n");
    //printf("\n");
  }

  fftw_execute(P[4]);
  fftw_execute(P[5]);

  double error =0.;

#pragma omp parallel for
  for (ptrdiff_t i =0; i < local_n[0]; i++){
    for (ptrdiff_t j = 0; j <ns[1]; j++){
      for (ptrdiff_t k = 0; k <ns[2]; k++){
        real_idx = (i*ns[1]+j)*2*(ns[2]/2+1) + k;
        //error += (physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j])*(physical[1][i*2*(ns[1]/2+1)+j] -physical[3][i*2*(ns[1]/2+1)+j]) ;
        //printf("%f \t", physical[2][real_idx]);
        error += (physical[0][real_idx] -physical[2][real_idx])*(physical[0][real_idx] -physical[2][real_idx]) ;
      }
      //printf("\n");
    }
    //printf("\n");
    //printf("\n");
  }

  //printf("Hello \n");
  printf("Time elapsed %f\n", MPI_Wtime()-tt);
  MPI_Barrier(MPI_COMM_WORLD);
  clean_data3D(physical, hatted, P);
  printf(" Error is %f \n", error);
}



void Fluid_solve_MPI(double **f, double **u, int nx, int ny, int nz, double mu){
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  const ptrdiff_t ns[3] = {nx, ny, nz};
  long Nex = 2*ns[1]*(ns[2]/2+1);
  const double Ls[3] = {1, 1, 1};
  ptrdiff_t real_idx, f_idx;
  fftw_plan P[NUM_PLANS3];

  double *physical[7];
  fftw_complex  *hatted[7];
  
  ptrdiff_t local_n[4];
  init_data3D(physical, hatted, ns, local_n, P);

  //printf("Fluid solve init successful, rank = %d, p = %d\n", rank, p); 
  ptrdiff_t *global_idxs = (ptrdiff_t*)malloc(2*p*sizeof(ptrdiff_t)); 
  int *global_ns = (int*)malloc(p*sizeof(int)); 
  int *global_diss = (int*)malloc(p*sizeof(int)); 

  MPI_Datatype FFTW_MPI_PTRDIFF_T;
  if(sizeof(ptrdiff_t) == sizeof(long)) {
    FFTW_MPI_PTRDIFF_T = MPI_LONG;
  }else if(sizeof(ptrdiff_t) == sizeof(long long)) {
    FFTW_MPI_PTRDIFF_T = MPI_LONG_LONG;
  }else if(sizeof(ptrdiff_t) == sizeof(int)) {
    FFTW_MPI_PTRDIFF_T = MPI_INT;
  } else {printf("Unknown ptrdiff_t type\n");}

  MPI_Gather(local_n, 2, FFTW_MPI_PTRDIFF_T, global_idxs, 2, FFTW_MPI_PTRDIFF_T,
      0, MPI_COMM_WORLD);

  //printf("Gather successful\n"); 
  if(rank==0){
    for(int i=0; i<p; i++){
      global_ns[i]   =global_idxs[2*i]*Nex;
      global_diss[i] =global_idxs[2*i+1]*Nex;
      //printf("n = %d, dis = %d\n", global_ns[i], global_diss[i]);
      //printf("n = %d, dis = %d\n", global_idxs[2*i]*ns[1]*ns[2],global_idxs[2*i+1]*ns[1]*ns[2]);
      //printf("n = %d, dis = %d\n", local_n[0], local_n[1]);
    }  
  }
  //printf("Extraction successful\n"); 
  for(int i=0; i<3; i++){
    //printf("Sending i = %d\n", i);
    MPI_Scatterv(f[i], global_ns, global_diss, MPI_DOUBLE, physical[4+i],
        local_n[0]*Nex, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  //printf("Scatter successful\n"); 
  fluid_solve_3D_mpi(physical, hatted, ns, Ls, mu, local_n, P);
     
  //print_array3D(physical[3], nx, ny, nz);
  print_hat_array3D(hatted[4], ns);
  print_hat_array3D(hatted[5], ns);
  print_hat_array3D(hatted[6], ns);
  print_hat_array3D(hatted[3], ns);
  //printf("Solve successful\n"); 
  for(int i=0; i<4; i++){
    MPI_Gatherv(physical[i],local_n[0]*Nex, MPI_DOUBLE, u[i], global_ns, global_diss, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  
  //printf("Gather successful\n"); 
  MPI_Barrier(MPI_COMM_WORLD);

  clean_data3D(physical, hatted, P);
  free(global_idxs);
  free(global_ns);
  free(global_diss);
}

void Fluid_test(int n){
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  long nx=n, ny=n, nz=n;
  long N = nx*ny*2*(nz/2+1);
  double *f[3], *u[4], *u_True[4];

  long idx;
  double tpi = 2*M_PI, x, y, z;
  double mu = .0;
  if(rank==0){
    for(int i=0; i<3; i++){
      f[i] = fftw_alloc_real(N);
      u[i] = fftw_alloc_real(N);
      u_True[i] = fftw_alloc_real(N);
    }
    u[3] = fftw_alloc_real(N);
    u_True[3] = fftw_alloc_real(N);
   
    printf("Allocation successful\n"); 
    for(long i=0; i<nx; i++){
      for(long j=0; j<ny; j++){
        for(long k=0; k<nz; k++){
          idx = (i*ny+j)*2*(nz/2+1) +k;
          x = (double)i/nx; y = (double)j/ny; z = (double)k/nz;
          x = tpi*x; y=tpi*y;z = tpi*z;
          f[0][idx] = tpi*(3*mu*tpi*cos(y)*sin(x)+cos(x)*sin(y))*sin(z);
          f[1][idx] = tpi*(cos(y)*sin(x)-1.5*tpi*mu*cos(x)*sin(y))*sin(z);
          f[2][idx] = tpi*(1.5*mu*tpi*cos(y)*cos(x)+sin(x)*sin(y))*cos(z);
           
          u[0][idx]=0.;u[1][idx]=0.;u[2][idx]=0.;u[3][idx]=0.;
          
          u_True[0][idx] =      sin(x)*cos(y)*sin(z);         
          u_True[1][idx] = -0.5*cos(x)*sin(y)*sin(z);         
          u_True[2][idx] = 0.5* cos(x)*cos(y)*cos(z);         
          u_True[3][idx] =      sin(x)*sin(y)*sin(z); 
        }
      }
    }
  }
  
  double tt = MPI_Wtime();
  printf("Initialization successful\n"); 
  Fluid_solve_MPI(f, u, nx, ny, nz, mu);

  printf("Solve successful, time ellapsed = %f\n", MPI_Wtime()-tt); 
  if(rank==0){//Compare and Clean-up
    double erroru=0, errorv=0, errorw=0, errorp=0;

    //print_array3D(f[0], nx, ny, nz);
    print_array3D(f[0], nx, ny, nz);
    print_array3D(u[3], nx, ny, nz);
    print_array3D(u_True[3], nx, ny, nz);

    for(long i=0; i<nx; i++){
      for(long j=0; j<ny; j++){
        for(long k=0; k<nz; k++){
          idx = (i*ny+j)*2*(nz/2+1) +k;
          erroru += abs(u[0][idx]-u_True[0][idx]);
          errorv += abs(u[1][idx]-u_True[1][idx]);
          errorw += abs(u[2][idx]-u_True[2][idx]);
          errorp += abs(u[3][idx]-u_True[3][idx]);
        }
      }
    }
    printf("u error = %f, v error = %f, w error = %f, p error = %f\n", erroru/nx/ny/nz,
        errorv/nx/ny/nz, errorw/nx/ny/nz, errorp/nx/ny/nz); 
    for(int i=0;i<3; i++){
      free(f[i]);
      free(u[i]);
      free(u_True[i]);
    }
    free(u[3]);
    free(u_True[3]);
  }
}


int main(int argc, char **argv){
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  threads_ok = provided >=MPI_THREAD_FUNNELED;
  if (threads_ok) threads_ok = fftw_init_threads();
  int fftw_mpi_init(void);

  if (threads_ok) fftw_plan_with_nthreads(omp_get_max_threads());

  //threeD_Example();
  Fluid_test(8);
  MPI_Finalize();
}

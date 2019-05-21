Parallel Implementation of the Immersed Boundary Method in 3D.

How to build:
Clone the git repository. Inside the ibmP folder, create a new directory called build. cd into build. Then run "../buildOMP.sh" to generate the makefile. Then type "make". This builds the OpenMP version of our code. To build the MPI version, FFTW3 must be installed with enable-mpi flag. The path also may need to be specified in the CMakeLists.txt in the ibmP directory. 

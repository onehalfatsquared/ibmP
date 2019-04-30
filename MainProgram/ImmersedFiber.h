//
//  ImmersedFiber.h
//  IBMethod3D
//
//  Created by Wanda Strychalski on 6/13/14.
//
//

class ImmersedFiber {
public:
    	ImmersedFiber(double r, double k, double a, int N);
	void calcForces();
	void spreadForces(double *forceX, double *forceY, double *forceZ, double* xEpts, 
   		int Nx, double* yEpts, int Ny, double* zEpts, int Nz);
	void updateBoundary();

private:
	double _r, _k, _a, _h;
	int _NIB;
	double *_xIB;
	double *_yIB;
	double *_zIB;
	int *_first, *_next; // linked lists for the bins
	double *_xForce, *_yForce, *_zForce;
	void binPoints(double he, double ax, double bx, double ay, double by, int Nx);
    
};

extern double phi(double r);
extern double modp(double x, double L);


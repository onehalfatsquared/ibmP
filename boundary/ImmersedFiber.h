

class ImmersedFiber {
public:
  ImmersedFiber(double L, double k, double kb, int N, int random);
  ~ImmersedFiber(); 
	void calcForces(int random);
	void spreadForces(double *forceX, double *forceY, double *forceZ, double* xEpts, 
 	int Nx, double* yEpts, int Ny, double* zEpts, int Nz);
	void getBoundaryVelocity(double *ugrid, double *vgrid, double *wgrid, double* xEpts, 
   int Nx, double* yEpts, int Ny, double* zEpts, int Nz);
	void updatePoints(double dt);

private:
	double _L, _k, _kb, _h;
	int _NIB;
	double *_xIB;
	double *_yIB;
	double *_zIB;
	double *_uIB;
	double *_vIB;
	double *_wIB;
	int *_first, *_next; // linked lists for the bins
	double *_xForce, *_yForce, *_zForce;
	void binPoints(double he, double ax, double bx, double ay, double by, int Nx);
    
};

extern double phi(double r);
extern double modp(double x, double L);


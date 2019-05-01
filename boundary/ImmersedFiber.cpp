#include "ImmersedFiber.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>
#include <omp.h>

const double eps = 1e-14;
/***************************************
 * Constructor
 *
 **************************************/
ImmersedFiber::ImmersedFiber(double r, double k, double a, int N){
    // r is the radius of the membrane / fiber at rest
    // k is the stiffness of the springs that connect each point
    // a is the radius of the ellipse
    // N is the number of IB points
	_r = r;
	_k = k;
	_a = a;
	_NIB = N;
	_h = 2.0*M_PI*_r/_NIB; //point spacing
	_xIB = (double*) malloc(_NIB*sizeof(double));
	_yIB = (double*) malloc(_NIB*sizeof(double));
	_zIB = (double*) malloc(_NIB*sizeof(double));
	_xForce = (double*) malloc(_NIB*sizeof(double));
	_yForce = (double*) malloc(_NIB*sizeof(double));
	_zForce = (double*) malloc(_NIB*sizeof(double));
	double theta;
	double b = _r/_a;
	for (int iPt = 0; iPt < _NIB; iPt++){
		theta=2.0*M_PI*iPt/((double)_NIB);
		_xIB[iPt]=_a*cos(theta);
		_yIB[iPt]=b*sin(theta);
		_zIB[iPt]=0;
		//printf("Point %d has (%f,%f,%f)\n",iPt,_xIB[iPt],_yIB[iPt],_zIB[iPt]);
	}
	
} // end constructor

//Deconstructor
ImmersedFiber::~ImmersedFiber() {
    free(_xIB); free(_yIB); free(_zIB);
    free(_xForce); free(_yForce); free(_zForce);
}



/***************************************
 * Find the elastic forces at the
 * current configuration
 ***************************************/
void ImmersedFiber::calcForces() {
	// Calculate the force due to fiber tension
	double tao11, tao12, tao13, tao21, tao22, tao23;
	double nt1, nt2, T1, T2;
	int indexm1, indexp1;
        for (int iPt=0; iPt < _NIB; iPt++){
		indexm1=iPt-1;
		indexp1=iPt+1;
		if (indexm1==-1) indexm1=_NIB-1;
		if (indexp1==_NIB) indexp1=0;
		tao11=(_xIB[iPt]-_xIB[indexm1])/_h;
		tao12=(_yIB[iPt]-_yIB[indexm1])/_h;
		tao13=(_zIB[iPt]-_zIB[indexm1])/_h;
		tao21=-(_xIB[iPt]-_xIB[indexp1])/_h;
		tao22=-(_yIB[iPt]-_yIB[indexp1])/_h;
		tao23=-(_zIB[iPt]-_zIB[indexp1])/_h;
		nt1 = sqrt(tao11*tao11+tao12*tao12+tao13*tao13);
		nt2 = sqrt(tao21*tao21+tao22*tao22+tao23*tao23);
		T1 = _k*(nt1-1.0);
		T2 = _k*(nt2-1.0);
		_xForce[iPt]=(T2*(tao21/nt2)-T1*(tao11/nt1));
		_yForce[iPt]=(T2*(tao22/nt2)-T1*(tao12/nt1));
		_zForce[iPt]=(T2*(tao23/nt2)-T1*(tao13/nt1));
		//printf("Point %d has (%f,%f,%f)\n",iPt,_xForce[iPt],_yForce[iPt],_zForce[iPt]);
	} // end loop over points
} // end of compute elastic forces

void ImmersedFiber::binPoints(double he, double ax, double bx, double ay, double by, int Nx){
    // Calculate the bins based on x and y
    double xImage, yImage;
    for (int iPt=0; iPt < _NIB; iPt++){
        xImage = (_xIB[iPt]-ax)/(bx-ax);
        xImage = _xIB[iPt] - ((double)(xImage > 0))*floor(abs(xImage));
        yImage = (_yIB[iPt]-ay)/(by-ay);
        yImage = _yIB[iPt] - ((double)(yImage > 0)*floor(abs(yImage)));
        int binNum=floor((xImage-ax)/he)+floor((yImage-ay)/he)*Nx;
        //cout << "Point " << iPt << " location " << xIB[iPt] << " , " << yIB[iPt] << endl;
        //cout << "Image: " << xImage << " , " << yImage << " and bin " << binNum << endl;
        if (_first[binNum]==-1){
            _first[binNum]=iPt;
        } else {
            int index=_first[binNum];
            int last;
            while (index > -1){
                last=index;
                index=_next[index];
            }
            _next[last]=iPt;
        }
    }
}
            


void ImmersedFiber::spreadForces(double *forceX, double *forceY, double *forceZ, double* xEpts, 
   int Nx, double* yEpts, int Ny, double* zEpts, int Nz){
    double hex=xEpts[1]-xEpts[0];
    double hey=yEpts[1]-yEpts[0];
    double hez=zEpts[1]-zEpts[0];
    double aex=xEpts[0];
    double bex=xEpts[Nx-1]+hex;
    double aey=yEpts[0];
    double bey=yEpts[Ny-1]+hey;
    double aez=zEpts[0];
    double bez=zEpts[Nz-1]+hez;
    double hm3 = 1.0/(hex*hey*hez);
    // Bin the points first to update arrays _first and _next
    // Reset _first and _next
    // Fill up the arrays with -1
    _first = (int*) malloc(Nx*Ny*sizeof(int));
    _next = (int*) malloc(_NIB*sizeof(int));
    for (int i=0; i< Nx*Ny; i++){
        _first[i]=-1;
    }
    for (int i=0; i< _NIB; i++){
        _next[i]=-1;
    }     
    binPoints(hex, aex, bex, aey, bey, Nx);
    // Zero out the force arrays
    for (int iPt=0; iPt < Nx*Ny*Nz; iPt++){
	forceX[iPt]=0; 
	forceY[iPt]=0;
	forceZ[iPt]=0;
    }
    // Loop over colors
    omp_set_num_threads(1);
    for (int color=0; color < 16; color++){
        #pragma omp parallel for schedule(dynamic)
        for (int yBin=(color/4)*Nx; yBin < Nx*Ny; yBin+=4*Nx){
            for (int xBin=color%4; xBin < Nx; xBin+=4){
            int bin=yBin+xBin;
            int ilam=_first[bin];
            while (ilam > -1){
	    //std::cout << "Updating point " << ilam << std::endl;
                int floorz=ceil((_zIB[ilam]-aez)/hez);
                int floory=ceil((_yIB[ilam]-aey)/hey);
                int floorx=ceil((_xIB[ilam]-aex)/hex);
                for (int z=floorz-2; z<floorz+2;z++){
                    int zIndex=((z%Nz)+Nz)%Nz;
                    double zw=phi(modp((zEpts[zIndex]-_zIB[ilam]),bez-aez)/hez);
                    for (int y=floory-2; y<floory+2;y++){
                        int yIndex=((y%Ny)+Ny)%Ny;
                        double yw=phi(modp((yEpts[yIndex]-_yIB[ilam]),bey-aey)/hey);
                        for (int x=floorx-2; x<floorx+2;x++){
                            int xIndex=((x%Nx)+Nx)%Nx;
                            double val=phi(modp((xEpts[xIndex]-_xIB[ilam]),bex-aex)/hex)*yw*zw;
                            int fluidIndex=zIndex*Ny*Nx+yIndex*Nx+xIndex;
                            forceX[fluidIndex]+=val*_xForce[ilam]*hm3;
                            forceY[fluidIndex]+=val*_yForce[ilam]*hm3;
                            forceZ[fluidIndex]+=val*_zForce[ilam]*hm3;
                        }
                    }
                }
                ilam = _next[ilam];
             }
            }
        }
    }
    free(_first);
    free(_next);
}


/***************************************
 * Update the boundary points with the local fluid velocity
 * Edit: surface boundary points
 **************************************/
 
void ImmersedFiber::updateBoundary() {
    

    
} // end update boundary




// Extern functions
/***************************************
 * Peskin delta function
 **************************************/
double phi(double r) {
    double val=0;
    if (abs(r) >=2){}
    else if (r <=-1){
        val=1.0/8.0*(5.0+2.0*r-sqrt(-7.0-12.0*r-4.0*r*r));}
    else if (r <=0){
        val=1.0/8.0*(3.0+2.0*r+sqrt(1.0-4.0*r-4.0*r*r));}
    else if (r <=1){
        val=1.0/8.0*(3.0-2.0*r+sqrt(1.0+4.0*r-4.0*r*r));}
    else if (r <=2){
        val=1.0/8.0*(5.0-2.0*r-sqrt(-7.0+12.0*r-4.0*r*r));}
    return val;
}

double modp(double x, double L){
    double p=x;
    if (x<-L/2){
        p=round(-x/L)*L+x;}
    else if (x>L/2){
        p=x-round(x/L)*L;}
    return p;
}
		

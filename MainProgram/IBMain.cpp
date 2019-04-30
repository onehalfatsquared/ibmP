#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include "ImmersedFiber.h"
#include <fstream>

//////////////////////////////////////////////////////////////////////////////
//    Main routine
//////////////////////////////////////////////////////////////////////////////

//void Computation();

int main(int argc,const char *argv[])
{
    // Input parameters. Will make into an input file later. 
    double sphereRadius = 1.0;
    double springStiff = 1.0;
    double ae = -1.0;
    double be = 1.0;
    int n = 32;
    double mu=1.0;
    double dt=1e-4;
    double until = 1.0;
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
    int NIB = (int)floor(4*M_PI*sphereRadius/h); // two points per meshwidth
    ImmersedFiber Fib = ImmersedFiber(sphereRadius, springStiff, 1.5, NIB);
    Fib.calcForces();
    Fib.spreadForces(gridFX, gridFY, gridFZ, xEpts, n, yEpts, n, zEpts, n);
    std::ofstream fout1("xGridForce.txt");
    std::ofstream fout2("yGridForce.txt");
    for (int iPt=0; iPt < n*n*n; iPt++){
	fout1 << gridFX[iPt] << std::endl;
	fout2 << gridFY[iPt] << std::endl;
    } 
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//    Computational routine
//////////////////////////////////////////////////////////////////////////////

/*void Computation(const DTMesh3DGrid &edgeGrid,double sphereRadius,
                 const DTPoint3D &sphereCenter,const string &surfaceFileName,
                 const DTDoubleArray &surfaceParameters,double mu,double dt,
                 double until,double saveEvery,
                 DTSeriesGroup<DT_RetGroup> &computed)
{
        // Insert your code here.
        
        DTProgress progress;
        DT_RetGroup returnStructure; // Fill this at each time value.
        int iteration = 0;
        
        int between = int(saveEvery);
        double time = 0;
        
        DTFile structureData(surfaceFileName,DTFile::ExistingReadWrite);
        // Set up the model
        SHSurface surfaceModel = SHSurface(sphereCenter,edgeGrid, surfaceParameters, structureData);
		surfaceModel.computeInterpolantandDerivs();
		surfaceModel.computeElasticForces();
        DTSurface3D surfaceEval = surfaceModel.getSurfaceEvalPts();
		DTSurface3D surfaceInterp = surfaceModel.getSurfaceInterpPts();
        DTVectorCollection3D elasticForceVectors = surfaceModel.getElasticForceVectors();
        DTMesh3D spreadForceUMesh = surfaceModel.getSpreadElasticForceUMesh();
        DTMesh3D spreadForceVMesh = surfaceModel.getSpreadElasticForceVMesh();
        DTMesh3D spreadForceWMesh = surfaceModel.getSpreadElasticForceWMesh();
        DTVectorCollection3D surfaceVelocity = surfaceModel.getBoundaryVelocity();
        // fluid solver
        StokesSolver3D fluidSolver = StokesSolver3D(edgeGrid, mu);
        DTMesh3D uMesh = fluidSolver.getUmesh();
        DTMesh3D vMesh = fluidSolver.getVmesh();
        DTMesh3D wMesh = fluidSolver.getWmesh();
        DTMesh3D pMesh = fluidSolver.getPmesh();
		returnStructure.surface = surfaceInterp;
        returnStructure.forceVectors = elasticForceVectors;
        returnStructure.boundaryVelocity = surfaceVelocity;
        returnStructure.fuMesh = spreadForceUMesh;
        returnStructure.fvMesh = spreadForceVMesh;
        returnStructure.fwMesh = spreadForceWMesh;
        returnStructure.uMesh = uMesh;
        returnStructure.vMesh = vMesh;
        returnStructure.wMesh = wMesh;
        returnStructure.pMesh = pMesh;
        computed.Add(returnStructure,time);
        
        // Inside the loop, do
        //     progress.UpdatePercentage(fraction);
        //     computed.Add(returnStructure,time); // Call with time>=0 and strictly increasing.
    double percentDone;
    percentDone = 0;
    
    //DTMatlabDataFile outFile = DTMatlabDataFile("/Users/wanda/Dropbox/UndergadResearch/Ondrej/OneTimeStepData.mat",DTFile::NewReadWrite);
    /*
     1) Lagrangian force
     2) Fluid force
     3) Fluid velocity
     4) Structure velocity
     */
    /*DTDoubleArray LagrangeForceData;
    DTDoubleArray fuMesh;
    DTDoubleArray fvMesh;
    DTDoubleArray fwMesh;
    DTDoubleArray structureVelocityData;
    DTDoubleArray uMeshData,vMeshData,wMeshData;
        while(time<until) {
            
            if (time+dt>=until-0.0001*dt) {
                // Last time step.
                dt = until-time;
                time = until;
                //	stop = true;
            }
            else {
                time+= dt;
            }
            //if(time >0.3) {
            //cerr<<time<<"\n";
            //	}
            iteration++;
			//surfaceModel.setOGBoundary();
			surfaceModel.computeInterpolantandDerivs();
            surfaceModel.computeElasticForces();
            surfaceModel.spreadForces();
            spreadForceUMesh = surfaceModel.getSpreadElasticForceUMesh();
            spreadForceVMesh = surfaceModel.getSpreadElasticForceVMesh();
            spreadForceWMesh = surfaceModel.getSpreadElasticForceWMesh();
           
            fluidSolver.fluidSolve(spreadForceUMesh,spreadForceVMesh,spreadForceWMesh);
            DTMesh3D uMesh = fluidSolver.getUmesh();
            DTMesh3D vMesh = fluidSolver.getVmesh();
            DTMesh3D wMesh = fluidSolver.getWmesh();
            surfaceModel.updateBoundary(uMesh, vMesh, wMesh,dt);
			/*surfaceModel.computeInterpolantandDerivs();
            surfaceModel.computeElasticForces();
            surfaceModel.spreadForces();
            spreadForceUMesh = surfaceModel.getSpreadElasticForceUMesh();
            spreadForceVMesh = surfaceModel.getSpreadElasticForceVMesh();
            spreadForceWMesh = surfaceModel.getSpreadElasticForceWMesh();
           
            fluidSolver.fluidSolve(spreadForceUMesh,spreadForceVMesh,spreadForceWMesh);
            DTMesh3D uMesh2 = fluidSolver.getUmesh();
            DTMesh3D vMesh2 = fluidSolver.getVmesh();
            DTMesh3D wMesh2 = fluidSolver.getWmesh();
			surfaceModel.updateOGBoundary(uMesh1, uMesh2, vMesh1, vMesh2, wMesh1, wMesh2, dt);*/
            /*if (iteration%between==0) {
                // save Data
                
                percentDone = 100.0*(time/until);
                cerr<<"Time is "<<time<<" ("<<percentDone<<"% done).\n";
                
				surfaceInterp = surfaceModel.getSurfaceInterpPts();
				returnStructure.surface = surfaceInterp;
                
                elasticForceVectors = surfaceModel.getElasticForceVectors();
                returnStructure.forceVectors = elasticForceVectors;
                
                surfaceVelocity = surfaceModel.getBoundaryVelocity();
                returnStructure.boundaryVelocity = surfaceVelocity;
                
                returnStructure.fuMesh = spreadForceUMesh;
                returnStructure.fvMesh = spreadForceVMesh;
                returnStructure.fwMesh = spreadForceWMesh;
                
                pMesh = fluidSolver.getPmesh();
                
                returnStructure.uMesh = uMesh;
                returnStructure.vMesh = vMesh;
                returnStructure.wMesh = wMesh;
                returnStructure.pMesh = pMesh;
                
                // savingFor Ondrej
                /*
                fuMesh = spreadForceUMesh.DoubleData();
                fvMesh = spreadForceVMesh.DoubleData();
                fwMesh = spreadForceWMesh.DoubleData();
                structureVelocityData = surfaceVelocity.Vectors();
                LagrangeForceData = elasticForceVectors.Vectors();
                uMeshData = uMesh.DoubleData();
                vMeshData = vMesh.DoubleData();
                wMeshData = wMesh.DoubleData();
                
             
                outFile.Save(fuMesh,"spreadForceUData");
                outFile.Save(fvMesh,"spreadForceVData");
                outFile.Save(fwMesh,"spreadForceWData");
                outFile.Save(uMeshData,"uMesh");
                outFile.Save(vMeshData,"vMesh");
                outFile.Save(wMeshData,"wMesh");
                
                outFile.Save(structureVelocityData,"membraneVelocity");
                outFile.Save(LagrangeForceData,"lagrangianForceUVWData");
                 */
                
                /*computed.Add(returnStructure,time);
            } // end if (iteration%between==0)
            progress.UpdatePercentage(time/until);
        } // end  while(time<until)

}*/

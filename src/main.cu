#include "pressure.h"
#include "matrix.h"
#include "dscr_sys.h"
#include "grid.h"
#include "helper.h"
#include "constants.h"
#include "stencil.h"
#include "solver.h"
#include "printvtk.h"
#include <iostream>
#include <vector>
#include <utility>

int main(){
    // Read and Move Config
    Config config("config.ini");
    setdConfigPressure(config);
    sethConfigPressure(config);
    setdConfigStencil(config);

    // Make grid on host
    LidDrivenCavity grid(config.DX, config.DY);

    // Make device versions
    DMatrix u(grid.u), v(grid.v), ustar(grid.u), vstar(grid.v), P(grid.P), u_ph(grid.u_ph), v_ph(grid.v_ph);
    
    // Storage for internal pressure points for pressure poisson solver
    std::vector<double> Pint((grid.P.nr-2)*(grid.P.nc-2),1);
    DVec dPint(Pint), dSource(Pint);

    // Make pressure poisson coefficient matrix
    DCSRSystem A(makePressurePoissonCoeff(grid.P.nr, grid.P.nc));
    int maxitr = grid.nx * grid.ny; // using max itr for cg solver based on grid size

    // Get Exec params, tpb: threads per block; nb : number of blocks
    auto [tpb, nb] = getExecParams();
    
    // Output filebase
    std::string filebase = "output/lid";
    std::string fileext = ".vtk";
    int wt = 0;
    int wT = (config.T/50) / config.dt;
    // Explicit Euler time loop
    int nT = config.T / config.dt;
    for (int t = 0; t < 2; ++t){
        
        // ToDo: Switch to 2 streams
        // Prediction Step
        ustarPred<<<nb, tpb>>>(ustar.Md, u.Md, v.Md);
        vstarPred<<<nb, tpb>>>(vstar.Md, u.Md, v.Md);
        applyustarBC<<<nb, tpb>>>(ustar.Md);
        applyvstarBC<<<nb, tpb>>>(vstar.Md);

        // Pressure Poisson Solve Step
        makeSourceTerm<<<nb, tpb>>>(ustar.Md, vstar.Md, P.Md, dSource.data);
        CGSolver(A, dPint, dSource, maxitr, 0.00001);
        setInternalPointsinP<<<nb, tpb>>>(P.Md, dPint.data);
        applyPressureBC<<<nb, tpb>>>(P.Md);

        // Correction Step
        uCorrect<<<nb, tpb>>>(u.Md, ustar.Md, P.Md);
        vCorrect<<<nb, tpb>>>(v.Md, vstar.Md, P.Md);
        applywBC<<<nb, tpb>>>(u.Md, ustar.Md);
        applywBC<<<nb, tpb>>>(v.Md, vstar.Md);

        if (t % wT == 0)
        {
        // Move to physical grid
        uphysical<<<nb, tpb>>>(u_ph.Md, u.Md);
        vphysical<<<nb, tpb>>>(v_ph.Md, v.Md);

        // Move to host
        u_ph.deviceMatrixToHost(grid.u_ph);
        v_ph.deviceMatrixToHost(grid.v_ph);

        std::string filename = filebase + "_t" + std::to_string(wt) + fileext;
        writevtk(grid.u_ph, grid.v_ph, filename);
        ++wt;
        }
    }

    // Move to physical grid
    uphysical<<<nb, tpb>>>(u_ph.Md, u.Md);
    vphysical<<<nb, tpb>>>(v_ph.Md, v.Md);

    // Move to host
    u_ph.deviceMatrixToHost(grid.u_ph);
    v_ph.deviceMatrixToHost(grid.v_ph);

    // Write to vtk
    writevtk(grid.u_ph, grid.v_ph, "test.vtk");
}
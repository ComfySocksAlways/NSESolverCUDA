#pragma once
#include "matrix.h"
#include "nodetype.h"

/* Rectangular Staggered Grid */
struct RSGrid{
    // Num cells in x dir of Physical Grid, i.e Columns
    int nx;                 
    // Num cells in y dir of Physical Grid, i.e. Rows
    int ny;                 
    // Staggered u velocity
    Matrix<double> u;       
    // Staggered v velocity
    Matrix<double> v;       
    // Pressure
    Matrix<double> P;
    // Node Type
    Matrix<NodeType> NT;

    Matrix<double> u_ph;

    Matrix<double> v_ph;

    Matrix<double> p_ph;

    RSGrid(double dx, double dy) :  nx(static_cast<int>(1.0/dx)),
                                    ny(static_cast<int>(1.0/dy)),
                                    u(ny+2,nx+1,0.0),v(ny+1,nx+2,0.0),
                                    P(ny+2,nx+2,0.0),
                                    NT(ny+2,nx+2,NodeType::FLUID),
                                    u_ph(nx,ny),v_ph(nx,ny),p_ph(nx,ny) {}
    virtual void setBoundary() = 0;
};

struct LidDrivenCavity : RSGrid{
    LidDrivenCavity(double dx, double dy) : RSGrid(dx, dy) { setBoundary(); }
    void setBoundary() override;
};

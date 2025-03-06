#pragma once
#include "matrix.h"
#include "grid.h"
#include "dscr_sys.h"
#include "constants.h"
#include "helper.h"

void setdConfigPressure(const Config &c);
void sethConfigPressure(const Config &c);

// ToDo: Fill need to flip signs from MATLAB code to make A positive definite

/// @brief  Construct Pressure Poisson Coefficient Equation to solve for pressure implicitly
/// @param Ap 
/// @param Pnr (In) : Number of rows in Pressure grid including ghost
/// @param Pnc (In) : Number of cols in Pressure grid including ghost
/// @return 
__host__ Matrix<double> makePressurePoissonCoeff(int Pnr, int Pnc);

/// @brief Construct source term for Pressure Poisson Equation
/// @param ustar (In) : Buffered Matrix
/// @param vstar (In) : Buffered Matrix
/// @param P   (In) : Buffered Matrix
/// @param Pin (Out) : Source term Vector
/// @return 
__global__ void makeSourceTerm(double *ustar, double *vstar, double *P, double *Pin);

/// @brief Set internal points(non ghost) in Pressure Matrix
/// @param P (In) : Buffered Matrix
/// @param Pin (In) : Vector
/// @return 
__global__ void setInternalPointsinP(double *P, double *Pin);

__global__ void applyPressureBC(double *P);

__global__ void checkaccess();

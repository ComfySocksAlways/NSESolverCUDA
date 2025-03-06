#pragma once
#include "dvec.h"
#include "matrix.h"
#include "dscr_sys.h"
#include "helper.h"
#include <vector>
#include <utility>
#include <stdexcept>

struct Jacobi{
    // already D^-1
    DVec Dinv;
    DCSRSystem T;
    Jacobi(std::pair<std::vector<double>,Matrix<double>> pair) : Dinv(pair.first), T(pair.second) {}
    // void Smoother(DVec &x, DVec &b, int maxItr, double tol);
};

/// @brief Perform smoothening for maxItr; ToDo: Find L2 norm of x-x_old for exit with tol
/// @param J
/// @param x
/// @param b
/// @param maxItr
/// @param tol
void JacobiSmoother(Jacobi &J, DVec &x, DVec &b, int maxItr, double tol);

std::pair<std::vector<double>, Matrix<double>> makeJacobi(const Matrix<double> &A);

/// @brief Perform y = Dx, where D is stored as a vector
/// @param d 
/// @param x 
/// @param y 
/// @param N 
/// @return 
__global__ void DiagonalMV(double* d, double* x,double* y, int N);

__global__ void xpby_jacobi(double *x, double *y, double *z, double b, int N);
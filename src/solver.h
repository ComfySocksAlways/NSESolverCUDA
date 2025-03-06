#pragma once
#include "matrix.h"
#include "dscr_sys.h"
#include "dvec.h"
#include "helper.h"
#include <cuda_runtime.h>

struct Stream{
    cudaStream_t s;
    Stream() { checkErr(cudaStreamCreate(&s)); }
    ~Stream() { checkErr(cudaStreamDestroy(s)); }
};

/// @brief Solve for Ax=b
/// @param A 
/// @param x 
/// @param b 
/// @param maxItr 
/// @param tol 
void CGSolver(DCSRSystem &A, DVec &x, DVec &b, int maxItr, double tol);

/// @brief Do c = a/b
/// @param a 
/// @param b 
/// @param c 
/// @return 
__global__ void divide(double *a, double *b, double *c);

/// @brief Do a=b
/// @param a 
/// @param b 
/// @return 
__global__ void Copyab(double *a, double *b);

__global__ void xpby(double *x, double *y,double* z, double* b, int N);

__global__ void xmby(double *x, double *y, double *z, double *b, int N);

__global__ void dot(double *x, double *y, double *result, int N);
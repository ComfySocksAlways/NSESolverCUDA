#pragma once
#include "helper.h"
#include<stdio.h>
#include "constants.h"

typedef double (*OneIpStencil)(int, int, double *);
typedef double (*TwoIpStencil)(int, int, double *, double*);

/// @brief Set global device config object
/// @param c 
void setdConfigStencil(const Config &c);

/// @brief             Apply stencil operation to y using u and v
/// @param y           (Out) Avoids boundaries of y
/// @param u           (In)
/// @param v           (In)
/// @param stencil     (In) : Function describing stencil operation
/// @return 
__device__ void applyStencil(double *y, double *u, double *v, TwoIpStencil stencil);

__device__ void applyStencil(double *y, double *u, OneIpStencil stencil);

/// @brief Discretized Diffusion term from NSE using 2nd Order Central Differencing Scheme
/// @param i 
/// @param j 
/// @param ncu      (In) : Number of cols in u or v 
/// @param u        (In) : u or v velocity component
/// @return
__device__ double cal_DiffusionTerm(int i, int j, int ncu, double *u);

/// @brief TwoIpStencil Type function for ustar
/// @param i        (In)
/// @param j        (In)
/// @param u        (In) : u velocity
/// @param v        (In) : v velocity
/// @return
__device__ double ustarStencil(int i, int j, double *u, double *v);

/// @brief TwoIpStencil Type function for vstar
/// @param i 
/// @param j 
/// @param u        (In) : u velocity
/// @param v        (In) : v velocity
/// @return 
__device__ double vstarStencil(int i, int j, double *u, double *v);

/// @brief Kernel for Prediction step of ustar
/// @param ustar 
/// @param u 
/// @param v 
/// @return 
__global__ void ustarPred(double *ustar, double *u, double *v);

/// @brief Kernel for Prediction step of ustar
/// @param vstar 
/// @param u 
/// @param v 
/// @return 
__global__ void vstarPred(double *vstar, double *u, double *v);

/// @brief Set all elements of row 'r' of device Matrix 'x' to 'val'
/// @param x 
/// @param r 
/// @param val 
/// @return 
__device__ void setRow(double *x, int r, double val);

/// @brief Set all elements of column 'c' of device Matrix 'x' to 'val'
/// @param x 
/// @param c 
/// @param val 
/// @return 
__device__ void setCol(double *x, int c, double val);

__global__ void applyustarBC(double *ustar);

__global__ void applyvstarBC(double *vstar);

/// @brief TwoIpStencil Type function for u
/// @param i 
/// @param j 
/// @param ustar 
/// @param P 
/// @return 
__device__ double ustencil(int i, int j, double *ustar, double *P);

/// @brief TwoIpStencil Type function for v
/// @param i 
/// @param j 
/// @param vstar 
/// @param P 
/// @return 
__device__ double vstencil(int i, int j, double *vstar, double *P);

/// @brief Correction step for u
/// @param u 
/// @param ustar 
/// @param P 
/// @return 
__global__ void uCorrect(double *u, double *ustar, double *P);

/// @brief Correction step for v
/// @param v 
/// @param vstar 
/// @param P 
/// @return 
__global__ void vCorrect(double *v, double *vstar, double *P);

/// @brief Apply BC for u or v
/// @param w (In) : u or v
/// @param wstar (In) : ustar or vstar
/// @return 
__global__ void applywBC(double *w, double *wstar);

__device__ double uphysicalstencil(int i, int j, double *u);

__device__ double vphysicalstencil(int i, int j, double *v);

__global__ void uphysical(double *uphy, double *u);

__global__ void vphysical(double *vphy, double *v);
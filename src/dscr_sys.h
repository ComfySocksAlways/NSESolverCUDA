#pragma once
#include "matrix.h"
#include <cusparse_v2.h>
#include<vector>
#include <cstdlib> 
#include "helper.h"
#include "sparse.h"
#include "dvec.h"

// ToDo: Make this a blackbox, i.e. pass in x and b to this, this system can do either Ax and return b, or solve for x.
/// @brief Device CSR Matrix handling
struct DCSRSystem{
    double* dvals;
    int *drowptr;
    int* dcolptr;
    static cusparseHandle_t handle;
    static bool handleInitialized;
    cusparseConstSpMatDescr_t descr;
    void *buffer;
    // ToDo 1 : If the dense Matrix creation has to be skipped first to make the sparse ...
    // ...      Make a new constructor that does the possion matrix creation from scratch
    DCSRSystem() {};
    DCSRSystem(const Matrix<double> &A);
    ~DCSRSystem() {     cudaFree(dvals);
                        cudaFree(drowptr);
                        cudaFree(dcolptr);
                        cusparseDestroySpMat(descr); }

    /// @brief Perform y = alpha*Ax + beta*y once, i.e buffer created and destroyed inside
    /// @param x 
    /// @param y 
    void DCSRSpMV(DVec &x, DVec& y, double alpha, double beta);

    /// @brief Make buffer for y = alpha*Ax + beta*y
    /// @param x 
    /// @param y 
    void MakeBuffer(DVec &x, DVec &y, double alpha, double beta);

    void PreProcess(DVec &x, DVec &y, double alpha, double beta);
    
    /// @brief Perform y = alpha*Ax + beta*y
    /// @param x 
    /// @param y 
    void Axplusy(DVec &x, DVec &y, double alpha, double beta);

    void ClearBuffer();
};
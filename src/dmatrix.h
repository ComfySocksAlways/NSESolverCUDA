#pragma once
#include "matrix.h"
#include "helper.h"

//RAII for device matrix.
// don't change the existing matrix.h file, just write a new class that can be used with newwer tests.

struct DMatrix{

    // Data allocated on Device
    double* Md;

    // Meta data available on Host
    int nr;
    int nc;

    /// @brief Construct Dmatrix from Host Matrix
    /// @param M 
    DMatrix(const Matrix<double>& M);

    ~DMatrix() { cudaFree(Md); }

    void deviceMatrixToHost(Matrix<double> &M);

    private:
        DMatrix(const DMatrix &);
};
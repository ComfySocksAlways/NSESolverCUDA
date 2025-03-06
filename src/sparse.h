#pragma once
#include "matrix.h"
#include <cusparse_v2.h>
#include<iostream>
#include<vector>
#include <iomanip>
#include <cstdlib> 
#include "helper.h"

/// @brief Host CSR with 0 indexing
struct CSR{
    int nnz;
    std::vector<double> vals;
    std::vector<int> rowptr;
    std::vector<int> colptr;
    int nr;
    int nc;
    CSR(const Matrix<double>& M);
    void printCSR();
};

template <typename T>
void printArray(T *a, int N);






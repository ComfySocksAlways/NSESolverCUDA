#pragma once
#include <cusparse_v2.h>
#include<vector>
#include "helper.h"
#include "matrix.h"
#include "dmatrix.h"

/// @brief Device Vector
struct DVec{
    double *data;
    cusparseDnVecDescr_t vecdescr;
    int ndata;
    int nr;
    int nc;

    DVec(){};
    
    DVec(int N);

    DVec(const std::vector<double>& input);

    /// @brief Construct from Device Matrix, use very carefully
    /// @param input (In) : Data will be copied with the offset
    DVec(const DMatrix& input);


    /// @brief Copy Ctor
    /// @param other 
    DVec(const DVec &other);
    
    // ToDo: Probably will need a separate constructor for making just the internal points

    ~DVec() {   cudaFree(data);
                cusparseDestroyDnVec(vecdescr); }

    void DVecToHost(std::vector<double> &output);

};
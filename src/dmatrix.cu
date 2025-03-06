#include "dmatrix.h"

DMatrix::DMatrix(const Matrix<double> & M){
    size_t size = sizeof(double) * (M.nc * M.nr + 2);
    checkErr(cudaMalloc(&Md,size));
    double dnr = static_cast<double>(M.nr);
    double dnc = static_cast<double>(M.nc);
    cudaMemcpy(Md, &dnr, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Md + 1, &dnc, sizeof(double), cudaMemcpyHostToDevice);
    checkErr(cudaMemcpy(Md + 2, M.data,  M.nr * M.nc * sizeof(double) , cudaMemcpyHostToDevice));
    // Cpy Meta data
    nr = M.nr;
    nc = M.nc;
}


void DMatrix::deviceMatrixToHost(Matrix<double> &M){
    // cudaMemcpy(&nr, Md, sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&nc, Md + 1, sizeof(double), cudaMemcpyDeviceToHost);
    if((static_cast<int>(nr) == M.nr ) && static_cast<int>(nc) == M.nc)
        cudaMemcpy(M.data, Md + 2, M.nr * M.nc * sizeof(double), cudaMemcpyDeviceToHost);
    else
        std::cout << "Size mismatch, could not copy" << std::endl;
}
#include "dvec.h"

DVec::DVec(int N){
    checkErr(cudaMalloc((&data), sizeof(double) * N));
    checkErr(cudaMemset(data,0,sizeof(double) * N));
    cusparseCreateDnVec(&vecdescr, N, data, CUDA_R_64F);
    ndata = N;
    
    // Construction from vector doesn't have matrix meta data
    nr = -1;
    nc = -1; 
}

DVec::DVec(const std::vector<double>& input){
    checkErr(cudaMalloc((&data), sizeof(double) * input.size()));
    checkErr(cudaMemcpy(data, input.data(), sizeof(double) * input.size(), cudaMemcpyHostToDevice));
    cusparseCreateDnVec(&vecdescr, input.size(), data, CUDA_R_64F);
    ndata = input.size();
    
    // Construction from vector doesn't have matrix meta data
    nr = -1;
    nc = -1;
}

DVec::DVec(const DMatrix& input){
    cusparseCreateDnVec(&vecdescr,input.nc*input.nr,input.Md+2,CUDA_R_64F);
}

DVec::DVec(const DVec &other){
    checkErr(cudaMalloc((&data), sizeof(double) * other.ndata));
    checkErr(cudaMemcpy(data, other.data, sizeof(double) * other.ndata, cudaMemcpyDeviceToDevice));
    cusparseCreateDnVec(&vecdescr, other.ndata, data, CUDA_R_64F);
    ndata = other.ndata;
    nr = other.nr;
    nc = other.nc;
}


void DVec::DVecToHost(std::vector<double>& output){
    size_t size = sizeof(double) * ndata;
    output.reserve(size);
    checkErr(cudaMemcpy(output.data(), data, size, cudaMemcpyDeviceToHost));
}



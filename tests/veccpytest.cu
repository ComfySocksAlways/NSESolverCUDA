#include<vector>
#include<iostream>
#include "../src/helper.h"
__global__ void addone(double* d_buffer, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stridex){
        d_buffer[i] += 1;
    }
}


int main(){

    std::vector<double> hmem = {1.0, 2.0, 3.0};

    double *dmem;
    cudaMalloc(&dmem, sizeof(double) * hmem.size());
    std::cout << "Size of host Mem " << hmem.size() << std::endl;
    checkErr(cudaMemcpy(dmem, hmem.data(), sizeof(double) * hmem.size(), cudaMemcpyHostToDevice));
    addone <<< 1, 1 >>> (dmem,hmem.size());
    checkErr(cudaMemcpy(hmem.data(),dmem, sizeof(double) * hmem.size(), cudaMemcpyDeviceToHost));
    for(auto a : hmem)
        std::cout << a << " ";
    std::cout << std::endl;
}
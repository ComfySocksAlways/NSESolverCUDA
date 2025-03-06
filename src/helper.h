#pragma once
#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<nvfunctional>
#include<utility>
#define CHECK_CUSPARSE(func) { cusparseStatus_t status = (func); if (status != CUSPARSE_STATUS_SUCCESS) { std::cerr << "cuSPARSE error: " << status << std::endl;  }}
#define FLT(i,j,nc) ( (i) * nc + j + 2 )

typedef void (*OneDGridBody)(int);

inline cudaError_t checkErr(cudaError_t err){
    if(err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return err;
}

__device__ inline void gridstride(int limit, nvstd::function<void(int)> body) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < limit; i += stride) {
        body(i);
    }
}

inline std::pair<int,int> getExecParams(){    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    return std::make_pair(prop.warpSize, prop.multiProcessorCount * 32);
}
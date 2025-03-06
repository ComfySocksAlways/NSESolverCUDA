#include <stdio.h>
#include "../src/helper.h"
__global__ void test(double* a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx==0)
        printf("a=%f\n", *a);
}
__global__ void initialize(double *b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<5)
        b[idx] = idx;
}
int main(){
    double *a, *b;
    cudaMalloc(&a, sizeof(double));
    cudaMalloc(&b, sizeof(double) * 5);
    checkErr(cudaMemsetAsync(a, 5, sizeof(double)));
    initialize<<<1, 10>>>(b);
    test<<<1, 1>>>(a);
    cudaDeviceSynchronize();
}
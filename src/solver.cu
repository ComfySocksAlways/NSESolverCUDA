#include "solver.h"

void CGSolver(DCSRSystem &A, DVec &x, DVec &b, int maxItr, double tol){

    // Initializations
    double hrnew = 0 , hrold = 0;
    double *drold, *drnew, *dpAp, *dalpha, *dbeta;
    Stream stream1;
    size_t sd = sizeof(double);
    cudaMalloc(&drnew, sd);
    cudaMalloc(&drold, sd);
    cudaMalloc(&dpAp, sd);
    cudaMalloc(&dalpha, sd);
    cudaMalloc(&dbeta, sd);
    cudaMemsetAsync(drnew, 0, sd);
    cudaMemsetAsync(drold, 0, sd);
    cudaMemsetAsync(dpAp, 0, sd);
    cudaMemsetAsync(dalpha, 0, sd);
    cudaMemsetAsync(dbeta, 0, sd);

    // Device Params
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;
    size_t sharedMemorySize = threads_per_block * sizeof(double);

    /* CG Algorithm
        taking that cusparse runs on default stream
    */
    DVec &r = b;                              // Source term will be modified
    A.MakeBuffer(x, r, -1.0, 1.0);           // Making buffer only once since all vectors are same size
    A.Axplusy(x, r, -1.0, 1.0);             // Initial Residual
    DVec p(r), Ap(r);                      // Inititalize search direction with residual
    dot<<<number_of_blocks, threads_per_block, sharedMemorySize>>>(r.data, r.data, drold, r.ndata);
    cudaMemcpy(&hrold, drold, sd, cudaMemcpyDeviceToHost);
    if(hrold == 0)                      // exit if residual is already 0
        return;
    A.PreProcess(p, Ap, 1.0, 0.0);
    for (int i = 0; i < maxItr; ++i)
    {
        A.Axplusy(p, Ap, 1.0, 0.0); // New search dir

        // Find Stepsize
        dot<<<number_of_blocks, threads_per_block, sharedMemorySize>>>(p.data,Ap.data, dpAp, r.ndata);
        divide<<<1, 1>>>(drold, dpAp, dalpha);

        // Update residula vector
        xmby<<<number_of_blocks, threads_per_block>>>(r.data, Ap.data, r.data, dalpha, r.ndata);

        // Update solution vector
        cudaStreamSynchronize(stream1.s);
        xpby<<<number_of_blocks, threads_per_block, 0, stream1.s>>>(x.data, p.data, x.data, dalpha, x.ndata);

        // Compute new residual
        dot<<<number_of_blocks, threads_per_block, sharedMemorySize>>>(r.data, r.data, drnew, r.ndata);
        divide<<<1, 1>>>(drnew, drold, dbeta);

        // Update old residual
        Copyab<<<1,1>>>(drold, drnew);

        // Update Search dir
        xpby<<<number_of_blocks, threads_per_block>>>(r.data, p.data, p.data, dbeta, p.ndata);

        // copy to device for exit condition
        cudaMemcpy(&hrnew, drnew, sd, cudaMemcpyDeviceToHost);

        if( hrnew < tol ){
            break;
        }
    }

    cudaFree(drnew);
    cudaFree(drold);
    cudaFree(dpAp);
    cudaFree(dalpha);
    cudaFree(dbeta);
    A.ClearBuffer();
}

__global__ void divide(double *a, double *b, double *c){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx==0)
        *c = (*a) / (*b);
}

__global__ void Copyab(double *a, double *b){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx==0)
        (*a) = (*b);
}

__global__ void xpby(double *x, double *y,double* z, double* b, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
               { z[i] = x[i] + (*b) * y[i]; };
}

__global__ void xmby(double *x, double *y,double* z, double* b, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    { z[i] = x[i] - (*b) * y[i]; };
}

__global__ void dot(double *x, double *y, double *result, int N){

    // Store temporaily to each block
    extern __shared__ double cache[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheIdx = threadIdx.x;
    double temp = 0;
    for (int i = idx; i < N; i += stride){
        temp += x[idx] * y[idx];
        if(i==0){
            *result = 0;     // make a unique thread overwrite current value
        }
    }
    cache[cacheIdx] = temp;
    __syncthreads();

    // Reduce it by progressively writing to half portions of block
    int i = blockDim.x / 2;
    while(i!=0){
        if(cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    // Thread 0 of each block will atomically add to global
    if(cacheIdx==0)
        atomicAdd(result, cache[cacheIdx]);
}
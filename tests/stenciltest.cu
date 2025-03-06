#include "../src/matrix.h"
#include "../src/helper.h"
#include "../src/stencil.h"
#include<string>

__device__ double ustarStencil(int i, int j, double* u){
    int nc = static_cast<int> (u[1]);
    return u[FLT(i, j + 1, nc)] - 2 * u[FLT(i, j, nc)] + u[FLT(i, j - 1, nc)];
}

__device__ void applyStencil(double *y, double *u, OneIpStencil stencil){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int nr = static_cast<int> (u[0]);
    int nc = static_cast<int> (u[1]);
    for (int i = idx; i < nr; i += stridex)
        for (int j = idy; j < nc; j += stridey){
            bool bcheck = (i == 0 || i == nr - 1) || (j == 0 || j == nc - 1);
            if(!bcheck)
                y[FLT(i,j,nc)] = stencil(i, j, u);
        }
}

__global__ void ustarupdate(double *ustar, double *u){
    applyStencil(ustar, u, ustarStencil);
}

int main(int argc, char *argv[]){

    int nr = std::stoi(argv[1]);
    int nc = std::stoi(argv[2]);
    Matrix<double> m(nr, nc, 1.0);
    Matrix<double> ustar(nr, nc, 1.0);
    printM(m);

    double *m_buffer = deviceMatrix(m);
    double *ustar_b = deviceMatrix(ustar);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    ustarupdate<<<number_of_blocks, threads_per_block>>>(ustar_b, m_buffer);
    checkErr(cudaGetLastError());
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // deviceMatrixToHost(m_buffer, m);
    deviceMatrixToHost(ustar_b, ustar);

    // printM(m);
    printM(ustar);
    std::cout << "Time taken in ms: " << milliseconds << std::endl;

    cudaFree(m_buffer);
    cudaFree(ustar_b);
    return 0;

}

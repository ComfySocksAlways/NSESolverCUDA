#include "../src/solver.h"
#include "../src/dvec.h"
#include <vector>

int main()
{

    std::vector<double> hx = {1, 2, 3, 4, 5};
    std::vector<double> hy = {2, 1, 1, 1, 1};

    DVec Dx(hx), Dy(hy);

    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;
    size_t sharedMemorySize = threads_per_block * sizeof(double);

    double result = 0;
    double *hresult;
    hresult = &result;
    double *dresult;
    cudaMalloc(&dresult, sizeof(double));
    cudaMemcpy(dresult, hresult, sizeof(double), cudaMemcpyHostToDevice);
    dot<<<number_of_blocks, threads_per_block,sharedMemorySize>>>(Dx.data,Dy.data,dresult,hx.size());
    cudaDeviceSynchronize();
    cudaMemcpy(hresult, dresult, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << *hresult << std::endl;
}
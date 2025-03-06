#include "../src/stencil.h"
#include "../src/grid.h"
#include "../src/matrix.h"


int main(int argc, char *argv[]){

    LidDrivenCavity grid(DX, DY);

    double *u_d = deviceMatrix(grid.u);
    double *v_d = deviceMatrix(grid.v);
    double *ustar_d = deviceMatrix(grid.u);
    double *vstar_d = deviceMatrix(grid.v);

    // Execution Configuration Setup
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;

    // Streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    ustarPred<<<number_of_blocks, threads_per_block, 0, stream1>>>(ustar_d, u_d, v_d);

    vstarPred<<<number_of_blocks, threads_per_block, 0, stream2>>>(vstar_d, u_d, v_d);
    checkErr(cudaGetLastError());

    applyustarBC<<<number_of_blocks, threads_per_block, 0, stream1>>>(ustar_d);

    applyvstarBC<<<number_of_blocks, threads_per_block, 0, stream2>>>(vstar_d);

    cudaDeviceSynchronize();

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    Matrix<double> ustar(grid.ny + 2, grid.nx + 1);
    Matrix<double> vstar(grid.ny + 1, grid.nx + 2);

    deviceMatrixToHost(ustar_d, ustar);
    deviceMatrixToHost(vstar_d, vstar);
    printM(grid.u);
    std::cout << "**********" << std::endl;
    printM(grid.v);
    std::cout << "**********" << std::endl;
    printM(ustar);
    std::cout << "**********" << std::endl;
    printM(vstar);

    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(ustar_d);
    cudaFree(vstar_d);
}
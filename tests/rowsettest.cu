#include "../src/stencil.h"
#include "../src/matrix.h"
#include <string>

__global__ void applyBC(double* m){
    setRow(m, 0, 0.0);
    setCol(m, 2, 3.0);
}
int main(int argc, char *argv[]){
    int nr = std::stoi(argv[1]);
    int nc = std::stoi(argv[2]);
    Matrix<double> m(nr, nc, 1.0);

    double *m_buffer = deviceMatrix(m);
    printM(m);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;

    applyustarBC<<<number_of_blocks, threads_per_block>>>(m_buffer);
    cudaDeviceSynchronize();

    deviceMatrixToHost(m_buffer, m);
    printM(m);

    cudaFree(m_buffer);


}
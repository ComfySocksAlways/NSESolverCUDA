#include "../src/matrix.h"
#include "../src/helper.h"
#include <string>
#include "../src/dmatrix.h"

__global__ void addone(double* d_buffer){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int nr = static_cast<int> (d_buffer[0]);
    int nc = static_cast<int> (d_buffer[1]);
    for (int i = idx + 2; i < nr*nc + 2; i += stridex){
        d_buffer[i] += 1;
    }
}

__global__ void ustarupdate(double* u, double* ustar){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int nr = static_cast<int> (u[0]);
    int nc = static_cast<int> (u[1]);
    for (int i = idx; i < nr; i += stridex)
        for (int j = idy; j < nc; j += stridey){
            auto flat = [&nc](int i_, int j_)
            { return i_ * nc + j_ + 2; };
            bool bcheck = (i == 0 || i == nr - 1) || (j == 0 || j == nc - 1);
            if(!bcheck)
                ustar[flat(i, j)] = u[flat(i, j + 1)] - 2 * u[flat(i, j)] + u[flat(i + 1, j)];
        }
}


// ToDo any device wrapper function to do loop for each element.

int main(int argc, char *argv[]){

    int nr = std::stoi(argv[1]);
    int nc = std::stoi(argv[2]);
    Matrix<double> m(nr, nc, 1.0);
    Matrix<double> ustar(nr, nc, 1.0);
    printM(m);

    // double *m_buffer = deviceMatrix(m);
    // double *ustar_b = deviceMatrix(ustar);
    DMatrix dm(m), dustar(ustar);

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
    addone<<<number_of_blocks, threads_per_block>>>(dustar.Md);
    checkErr(cudaGetLastError());
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // deviceMatrixToHost(m_buffer, m);
    dustar.deviceMatrixToHost(ustar);

    // printM(m);
    printM(ustar);
    std::cout << "Time taken in ms: " << milliseconds << std::endl;

    return 0;
}
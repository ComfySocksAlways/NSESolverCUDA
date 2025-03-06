#include "../src/pressure.h"
#include "../src/matrix.h"
#include "../src/dscr_sys.h"
#include "../src/grid.h"
#include "../src/helper.h"
#include "../src/constants.h"
#include "../src/stencil.h"

__global__ void addone(double* d_buffer){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int nr = static_cast<int> (d_buffer[0]);
    int nc = static_cast<int> (d_buffer[1]);
    for (int i = idx + 2; i < nr*nc + 2; i += stridex){
        d_buffer[i] += 1;
    }
}

int main(){
    LidDrivenCavity grid(DX, DY);
    DMatrix u(grid.u), v(grid.v), P(grid.P), ustar(grid.u), vstar(grid.v);
    std::vector<double> Pinteral((grid.P.nr-2)*(grid.P.nc-2),0);
    DVec Pint(Pinteral);

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int threads_per_block = prop.warpSize;
    int number_of_blocks = prop.multiProcessorCount * 32;

    addone<<<number_of_blocks, threads_per_block>>>(P.Md);

    makeSourceTerm<<<number_of_blocks, threads_per_block>>>(u.Md, v.Md, P.Md, Pint.data);

    setInternalPointsinP<<<number_of_blocks, threads_per_block>>>(P.Md, Pint.data);

    applyPressureBC<<<number_of_blocks, threads_per_block>>>(P.Md);

    uCorrect<<<number_of_blocks, threads_per_block>>>(u.Md, ustar.Md, P.Md);

    v.deviceMatrixToHost(grid.v);
    addone<<<number_of_blocks, threads_per_block>>>(vstar.Md);
    Matrix<double> hustar(grid.u);
    Matrix<double> hvstar(grid.v);
    vstar.deviceMatrixToHost(hvstar);
    printM(hvstar);
    printM(grid.v);

    std::cout << "apply vBC" << std::endl;

    applywBC<<<number_of_blocks, threads_per_block>>>(v.Md, vstar.Md);
    v.deviceMatrixToHost(grid.v);
    printM(grid.v);
}
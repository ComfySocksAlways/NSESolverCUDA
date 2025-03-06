#include "jacobi.h"

void JacobiSmoother(Jacobi& J, DVec &x, DVec &b, int maxItr, double tol){

    // Device Params
    auto [tpb, nb] = getExecParams();

    // f = -T.x
    DVec f(x.ndata);         // required to avoid modifying b by cusparse functions
    DVec g(x.ndata);
    J.T.MakeBuffer(x, f, -1.0, 0.0);
    J.T.PreProcess(x, f, -1.0, 0.0);
    for (int i = 0; i < maxItr; ++i){
        J.T.Axplusy(x, f, -1.0, 0.0);

        xpby_jacobi<<<nb, tpb>>>(b.data, f.data, g.data, 1.0, x.ndata);

        DiagonalMV<<<nb, tpb>>>(J.Dinv.data, g.data, x.data, x.ndata);
    
    }
}


std::pair<std::vector<double>,Matrix<double>> makeJacobi(const Matrix<double>& A){
    std::vector<double> hd(A.nr);
    Matrix<double> hT(A);
    for (int i = 0; i < A.nr; ++i)
        for (int j = 0; j < A.nc; ++j){
            if(i==j){
                if(A(i,j)==0){
                    std::cerr << "Error: Division by zero at A(" << i << ", " << j << ")\n";
                    throw std::runtime_error("Division by zero encountered in matrix A.");
                }
                hd[i]=(1 / A(i, j));
                hT(i, j) = 0;
            }
        }
    return std::make_pair(hd, hT);
}


__global__ void DiagonalMV(double* d, double* x,double* y, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
        y[i] = d[i] * x[i];
}


__global__ void xpby_jacobi(double *x, double *y,double* z, double b, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    { z[i] = x[i] + (b) * y[i]; };
}

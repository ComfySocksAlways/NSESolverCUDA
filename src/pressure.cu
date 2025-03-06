#include "pressure.h"

__device__ Config dCon;
Config hCon;

void setdConfigPressure(const Config &c){
    cudaMemcpyToSymbol(dCon, &c, sizeof(Config));
}
void sethConfigPressure(const Config &c){
    hCon = c;
}

__host__ Matrix<double> makePressurePoissonCoeff(int Pnr, int Pnc){
    // We solve only for internal points, so number of internal rows(m) and cols(n)
    int m = Pnr - 2;
    int n = Pnc - 2;
    int N = m * n;
    Matrix<double> hA(N,N,0.0);
    // Done: Sign filled here
    double alpha = 2 * hCon.dt * (1 / (hCon.DX * hCon.DX) + 1 / (hCon.DY * hCon.DY));
    double betax = -hCon.dt / (hCon.DX * hCon.DX);
    double betay = -hCon.dt / (hCon.DY * hCon.DY);

    /*
        Upper sub diagonals, j>i, i.e. first upper subdiagonal i == j-1
    */
    for (int i = 0; i < N;++i){
        for (int j = 0; j < N; ++j){

            // Main Diagonal
            if(i==j)
                hA(i, j) = alpha;

            // First upper subdiagonal
            if(i==j-1){
                hA(i, j) = betax;
                if((i+1)%m==0)  // 0 indexing for i, m is size i.e. m=4rows, 4th row is i=3
                    hA(i, j) = 0;
            }

            // First lower subdiagonal
            if(j==i-1){
                hA(i, j) = betax;
                if((i-1+1)%m==0)        // 0 indexing for i
                    hA(i, j) = 0;
            }

            // mth upper and lowe diagonal
            if((i==j-n) || (j==i-m))
                hA(i, j) = betay;
            
        }
    }
    return hA;
}


__global__ void makeSourceTerm(double *ustar, double *vstar, double *P, double *Pin){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int Pnr = static_cast<int>(P[0]);
    int Pnc = static_cast<int>(P[1]);
    int unc = static_cast<int>(ustar[1]);
    int vnc = static_cast<int>(vstar[1]);
    double betax = -dCon.dt / (dCon.DX * dCon.DX);
    double betay = -dCon.dt / (dCon.DY * dCon.DY);

    // Done: Sign flipped from Matlab code
    for (int i = idx; i < Pnr; i += stridex)
        for (int j = idy; j < Pnc; j += stridey){
            bool gcheck = (i == 0 || i == Pnr - 1) || (j == 0 || j == Pnc - 1);       // Skip Ghost points
            if(!gcheck){
                int k = (i - 1) * (Pnc - 2) + (j - 1);      // Index to go through internal points
                Pin[k] = -((ustar[FLT(i, j, unc)] - ustar[FLT(i, j - 1, unc)]) / dCon.DX +
                                        (vstar[FLT(i, j, vnc)] - vstar[FLT(i - 1, j, vnc)]) / dCon.DY);
                if (j == 1) {  // Left Wall
                    Pin[k] += betax * P[FLT(i, j - 1, Pnc)];
                }
                if (i == 1) {  // Top Wall
                    Pin[k] += betay * P[FLT(i - 1, j, Pnc)];
                }
                if (j == Pnc - 2) {  // Right Wall
                    Pin[k] += betax * P[FLT(i, j + 1, Pnc)];
                }
                if (i == Pnr - 2) {  // Bottom Wall
                    Pin[k] += betay * P[FLT(i + 1, j, Pnc)];
                }
            }
        }
}


__global__ void setInternalPointsinP(double *P, double *Pin){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int Pnr = static_cast<int>(P[0]);
    int Pnc = static_cast<int>(P[1]);
    for (int i = idx; i < Pnr; i += stridex)
        for (int j = idy; j < Pnc; j += stridey){
            bool gcheck = (i == 0 || i == Pnr - 1) || (j == 0 || j == Pnc - 1);       // Skip Ghost points
            if(!gcheck){
            int k = (i - 1) * (Pnc - 2) + (j - 1);      // Index to go through internal points
            P[FLT(i, j, Pnc)] = Pin[k];
            }
        }
}

__global__ void applyPressureBC(double *P){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int Pnr = static_cast<int>(P[0]);
    int Pnc = static_cast<int>(P[1]);
    for (int i = idx; i < max(Pnr, Pnc); i+= stridex){
        if(i<Pnc){
            P[FLT(0, i, Pnc)] = P[FLT(1, i, Pnc)];                  // bottom wall
            P[FLT(Pnr - 1, i, Pnc)] = P[FLT(Pnr - 2, i, Pnc)];      // top wall
        }
        if(i<Pnr){
            P[FLT(i, 0, Pnc)] = P[FLT(i, 1, Pnc)];                  // left wall
            P[FLT(i, Pnc - 1 , Pnc)] = P[FLT(i, Pnc - 2, Pnc)];     // right wall
        }
    }
    // int idy = threadIdx.y + blockDim.y * blockIdx.y;
    // int stridey = blockDim.y * gridDim.y;
    // int Pnr = static_cast<int>(P[0]);
    // int Pnc = static_cast<int>(P[1]);
    // for (int i = idx; i < Pnr; i += stridex)
    //     for (int j = idy; j < Pnc; j += stridey){
    //         if(i == 0)
    //             P[FLT(0, j, Pnc)] = P[FLT(1, j, Pnc)];
    //         if(i == Pnr - 1)
    //             P[FLT(Pnr - 1, j, Pnc)] = P[FLT(Pnr - 2, j, Pnc)];
    //         if(j == 0)
    //             P[FLT(i, 0, Pnc)] = P[FLT(1, 1, Pnc)];
    //         if(j == Pnc - 1)
    //             P[FLT(i, Pnc - 1 , Pnc)] = P[FLT(1, Pnc - 2, Pnc)];
    //     }
}


__global__ void checkaccess(){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx==0){
        printf("%f\n",dCon.dt);
    }
}
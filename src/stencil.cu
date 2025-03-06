#include "stencil.h"

__device__ Config dCon;
void setdConfigStencil(const Config &c){
    cudaMemcpyToSymbol(dCon, &c, sizeof(Config));
}

__device__ void applyStencil(double *y, double *u, double *v, TwoIpStencil stencil){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int nr = static_cast<int>(y[0]);
    int nc = static_cast<int>(y[1]);
    for (int i = idx; i < nr; i += stridex)
        for (int j = idy; j < nc; j += stridey){
            bool bcheck = (i == 0 || i == nr - 1) || (j == 0 || j == nc - 1);       // ToDo: Switch to Enum, but i think each field will require it's own enum
            if(!bcheck)
                y[FLT(i,j,nc)] = stencil(i, j, u, v);
        }
}

__device__ void applyStencil(double *y, double *u, OneIpStencil stencil){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int stridey = blockDim.y * gridDim.y;
    int nr = static_cast<int>(y[0]);
    int nc = static_cast<int>(y[1]);
    for (int i = idx; i < nr; i += stridex)
        for (int j = idy; j < nc; j += stridey){
            y[FLT(i,j,nc)] = stencil(i, j, u);
        }
}

__device__  double cal_DiffusionTerm(int i, int j, int ncu, double *u) {  
    return (u[FLT(i, j + 1, ncu)] - 2 * u[FLT(i, j, ncu)] + u[FLT(i, j - 1, ncu)]) / (dCon.DX * dCon.DX) +
           (u[FLT(i + 1, j, ncu)] - 2 * u[FLT(i, j, ncu)] + u[FLT(i - 1, j, ncu)]) / (dCon.DY * dCon.DY);   
}

__device__ double ustarStencil(int i, int j, double *u, double *v){

    int ncu = static_cast<int>(u[1]);
    int ncv = static_cast<int>(v[1]);

    // Diffusion
    double diffusion_term = dCon.mu * cal_DiffusionTerm(i, j, ncu, u);


    // Convection
    double u_east = 0.5 * (u[FLT(i, j+1, ncu)] + u[FLT(i, j, ncu)]);
    double u_west = 0.5 * (u[FLT(i, j-1, ncu)] + u[FLT(i, j, ncu)]);
    double u_north = 0.5 * (u[FLT(i+1, j, ncu)] + u[FLT(i, j, ncu)]);
    double u_south = 0.5 * (u[FLT(i-1, j, ncu)] + u[FLT(i, j, ncu)]);
    double v_north = 0.5 * (v[FLT(i, j+1, ncv)] + v[FLT(i, j, ncv)]);
    double v_south = 0.5 * (v[FLT(i-1, j+1, ncv)] + v[FLT(i-1, j, ncv)]);
    double convec_term = ((u_east * u_east) - (u_west * u_west)) / dCon.DX + 
                         (u_north * v_north - u_south * v_south) / dCon.DY;
    return u[FLT(i, j, ncu)] + dCon.dt * (diffusion_term - convec_term);
}


__device__ double vstarStencil(int i, int j, double *u, double *v){
    
    int ncu = static_cast<int>(u[1]);
    int ncv = static_cast<int>(v[1]);
    
    // Diffusion
    double diffusion_term = dCon.mu * cal_DiffusionTerm(i, j,ncv, v);

    // Convection
    double v_east = 0.5 * (v[FLT(i, j+1, ncv)] + v[FLT(i, j, ncv)]);
    double v_west = 0.5 * (v[FLT(i, j-1, ncv)] + v[FLT(i, j, ncv)]);
    double v_north = 0.5 * (v[FLT(i+1, j, ncv)] + v[FLT(i, j, ncv)]);
    double v_south = 0.5 * (v[FLT(i-1, j, ncv)] + v[FLT(i, j, ncv)]);
    double u_east = 0.5 * (u[FLT(i+1, j, ncu)] + u[FLT(i, j, ncu)]);
    double u_west = 0.5 * (u[FLT(i+1, j-1, ncu)] + u[FLT(i, j-1, ncu)]);
    double convec_term = (u_east * v_east - u_west * v_west) / dCon.DX + 
                         ((v_north * v_north) - (v_south * v_south)) / dCon.DY;
    return v[FLT(i, j, ncv)] + dCon.dt * (diffusion_term - convec_term);
}

__global__ void ustarPred(double *ustar, double *u, double *v){
    applyStencil(ustar, u, v, ustarStencil);
}

__global__ void vstarPred(double *vstar, double *u, double *v){
    applyStencil(vstar, u, v, vstarStencil);
}

__device__ void setRow(double *x, int r, double val){
    int nc = static_cast<int>(x[1]);
    gridstride(nc, [&](int j)
               { x[FLT(r, j, nc)] = val; });
}

__device__ void setCol(double *x, int c, double val){
    int nr = static_cast<int>(x[0]);
    int nc = static_cast<int>(x[1]);
    gridstride(nr, [&](int i)
               { x[FLT(i, c, nc)] = val; });
}

__global__ void applyustarBC(double *ustar){
    int nr = static_cast<int>(ustar[0]);
    int nc = static_cast<int>(ustar[1]);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = idx; j < nc; j += stride){ 
                ustar[FLT(0, j, nc)] = -ustar[FLT(1, j, nc)];
                ustar[FLT((nr-1),j,nc)] = 2.0 - ustar[FLT((nr-2),j,nc)]; };
}

__global__ void applyvstarBC(double *vstar){
    int nr = static_cast<int>(vstar[0]);
    int nc = static_cast<int>(vstar[1]);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < nr; i += stride){ 
        vstar[FLT(i, 0, nc)] = -vstar[FLT(i, 1, nc)];
        vstar[FLT(i,(nc-1),nc)] = -vstar[FLT(i,(nc-2),nc)]; };
}

__device__ double ustencil(int i, int j, double* ustar, double* P){
    int ncu = static_cast<int>(ustar[1]);
    int ncP = static_cast<int>(P[1]);
    return ustar[FLT(i, j, ncu)] - (dCon.dt / dCon.DX) * (P[FLT(i, j + 1, ncP)] - P[FLT(i, j, ncP)]);
}

__device__ double vstencil(int i, int j, double* vstar, double* P){
    int ncv = static_cast<int>(vstar[1]);
    int ncP = static_cast<int>(P[1]);
    return vstar[FLT(i,j,ncv)] - (dCon.dt / dCon.DY) * (P[FLT(i + 1, j, ncP)] - P[FLT(i, j, ncP)]);
}

__global__ void uCorrect(double* u, double* ustar, double* P){
    applyStencil(u, ustar, P, ustencil);
}

__global__ void vCorrect(double* v, double* vstar, double* P){
    applyStencil(v, vstar, P, vstencil);
}

__global__ void applywBC(double *w, double *wstar){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stridex = blockDim.x * gridDim.x;
    int unr = static_cast<int>(w[0]);
    int unc = static_cast<int>(w[1]);
    for (int i = idx; i < max(unr, unc); i+= stridex){
        if(i<unc){
            w[FLT(0, i, unc)] = wstar[FLT(0, i, unc)];              // Bottom
            w[FLT(unr-1, i, unc)] = wstar[FLT(unr-1, i, unc)];      // Top 
        }
        if(i<unr){
            w[FLT(i, 0, unc)] = wstar[FLT(i, 0, unc)];              // Left
            w[FLT(i, unc - 1, unc)] = wstar[FLT(i, unc - 1, unc)];  // Right
        }
    }
}

__device__ double uphysicalstencil(int i, int j, double* u){
    int ncu = static_cast<int>(u[1]);
    return 0.5 * (u[FLT(i, j, ncu)] + u[FLT(i + 1, j, ncu)]);
}

__device__ double vphysicalstencil(int i, int j, double* v){
    int ncv = static_cast<int>(v[1]);
    return 0.5 * (v[FLT(i, j, ncv)] + v[FLT(i, j + 1, ncv)]);
}

__global__ void uphysical(double* uphy, double* u){
    applyStencil(uphy, u, uphysicalstencil);
}

__global__ void vphysical(double* vphy, double* v){
    applyStencil(vphy, v, uphysicalstencil);
}

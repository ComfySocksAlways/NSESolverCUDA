#include "matrix.h"

template <typename T>
void printM(const Matrix<T>& M){
    // std::streamsize defaultprecision = std::cout.precision();
    if constexpr (std::is_same_v<T,double>)
        std::cout<<std::setprecision(3)<<std::fixed;
    else if constexpr (std::is_same_v<T,int>)
        std::cout<<std::setprecision(0)<<std::fixed;
    int plimit = 10;
    if ( max(M.nr,M.nc) <= plimit ){
        for (int i = 0; i < M.nr; ++i){
            for (int j = 0; j < M.nc; ++j){
                std::cout << M(i,j) << " ";
            }
            std::cout << std::endl; } }
    else{
        plimit = min(min(M.nr, M.nc), plimit);
         for (int i = 0; i < plimit; ++i){
            int i_o = (i >= plimit / 2) ? (M.nr - plimit) : 0;
            for (int j = 0; j < plimit; ++j) {
                int j_o = (j >= plimit / 2) ? (M.nc - plimit) : 0;
                std::cout << M(i+i_o, j+j_o) << " ";
                if(j==plimit/2 && M.nc > plimit) {std::cout << "... ";}
            }
            std::cout << std::endl;
            if(i==plimit/2 && M.nr > plimit) {std::cout << ".\n.\n."<<std::endl;}
             } }
    std::cout<<std::setprecision(2)<<std::fixed;
}

template void printM<double>(const Matrix<double>&);    // required to avoid linker error.
template void printM<NodeType>(const Matrix<NodeType> &);

template <typename T>
T *deviceMatrix(const Matrix<T> &M){
    T *Md;
    size_t size = sizeof(T) * (M.nc * M.nr + 2);
    checkErr(cudaMalloc(&Md,size));
    T nr = static_cast<T>(M.nr);
    T nc = static_cast<T>(M.nc);
    cudaMemcpy(Md, &nr, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(Md + 1, &nc, sizeof(T), cudaMemcpyHostToDevice);
    checkErr(cudaMemcpy(Md + 2, M.data,  M.nr * M.nc * sizeof(T) , cudaMemcpyHostToDevice));
    return Md;
}
template double *deviceMatrix(const Matrix<double> &M);     // required to avoid linker error
template NodeType *deviceMatrix(const Matrix<NodeType> &M);     // required to avoid linker error

template <typename T>
void deviceMatrixToHost(T *d_buffer, Matrix<T> &M){
    T nr, nc;
    cudaMemcpy(&nr, d_buffer, sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nc, d_buffer + 1, sizeof(T), cudaMemcpyDeviceToHost);
    if((static_cast<int>(nr) == M.nr ) && static_cast<int>(nc) == M.nc)
        cudaMemcpy(M.data, d_buffer + 2, M.nr * M.nc * sizeof(T), cudaMemcpyDeviceToHost);
    else
        std::cout << "Size mismatch, could not copy" << std::endl;
}
template void deviceMatrixToHost(double *d_buffer, Matrix<double> &M);   // required to avoid linker error
template void deviceMatrixToHost(NodeType *d_buffer, Matrix<NodeType> &M);   // required to avoid linker error



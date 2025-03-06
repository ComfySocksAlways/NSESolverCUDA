#include "sparse.h"


CSR::CSR(const Matrix<double>& M){
    // Alloc Estimate Sizes
    rowptr.resize(M.nr + 1, 0);

    // Store Values
    nnz = 0;
    for (int i = 0; i < M.nr; ++i){
        for (int j = 0; j < M.nc; ++j){
            if(M(i,j)!=0){
                vals.push_back(M(i, j));
                colptr.push_back(j);
                nnz++;
            }
        }
        rowptr[i + 1] = nnz;
    }
    // Store Original Shape information
    nr = M.nr;
    nc = M.nc;
};


template<typename T>
void printArray(T* a, int N){
    for (int i = 0; i < N;++i){
        std::cout << a[i] << " ";
    }
}
template void printArray<double>(double* a, int N);
template void printArray<int>(int* a, int N);

void CSR::printCSR(){
    std::streamsize defaultprecision = std::cout.precision();
    std::cout<<std::setprecision(2)<<std::scientific;
    int plimit = 7;
    if (nnz <= plimit){
        std::cout<<"Vals: [ ";
        printArray(vals.data(), nnz);
        std::cout << "]" << std::endl;
        std::cout<<"ColPtr: [ ";
        printArray(colptr.data(), nnz);
        std::cout << "]" << std::endl;
        std::cout<<"RowPtr: [ ";
        printArray(rowptr.data(), nr+1);
        std::cout << "]" << std::endl;
    }
    else{
        std::cout<<"Val: [ ";
        printArray(vals.data(), plimit);
        std::cout << "... " << std::endl;
        printArray(vals.data()+nnz-plimit, plimit);
        std::cout << "]" << std::endl;
    }

    std::cout.precision(defaultprecision);
}

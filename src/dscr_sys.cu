#include "dscr_sys.h"


cusparseHandle_t DCSRSystem::handle;
bool DCSRSystem::handleInitialized = false;
// DCSRSystem::DCSRSystem(){

// }

DCSRSystem::DCSRSystem(const Matrix<double> &A){
    CSR csrM(A);
    // Coeff Matrix Alloc
    checkErr(cudaMalloc(&dvals, sizeof(double) * csrM.nnz));
    checkErr(cudaMalloc(&dcolptr, sizeof(int) * csrM.nnz));
    checkErr(cudaMalloc(&drowptr, sizeof(int) * (csrM.nr+1)));

    // Copy Data
    checkErr(cudaMemcpy(dvals, csrM.vals.data(), sizeof(double) * csrM.nnz, cudaMemcpyHostToDevice));
    checkErr(cudaMemcpy(dcolptr, csrM.colptr.data(), sizeof(int) * csrM.nnz, cudaMemcpyHostToDevice));
    checkErr(cudaMemcpy(drowptr, csrM.rowptr.data(), sizeof(int) * (csrM.nr+1), cudaMemcpyHostToDevice));

    // cuSparse Handles
    if(!handleInitialized){
        cusparseCreate(&handle);
        handleInitialized = true;
        std::atexit([]() {
            cusparseDestroy(handle);
        });
    }
    cusparseCreateConstCsr(     &descr,
                                csrM.nr,
                                csrM.nc,
                                csrM.nnz,
                                drowptr,
                                dcolptr,
                                dvals,
                                CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_64F
                        );
}   

void DCSRSystem::DCSRSpMV(DVec &x, DVec& y, double alpha, double beta){
 size_t bufferSize;
 CHECK_CUSPARSE(   cusparseSpMV_bufferSize(handle,
                            cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            descr,
                            x.vecdescr,
                            &beta,
                            y.vecdescr,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG1,
                            &bufferSize
                            ));
    void *externalBuffer;
    cudaMalloc(&externalBuffer, bufferSize);
    // ToDo 1: Ensure this preprocess is only called once, for multi calls to this product.
    // ToDo 2: Maybe make x of const type?
    // cusparseSpMV_preprocess(handle,
    //                         cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         &alpha,
    //                         descr,
    //                         x.vecdescr,
    //                         &beta,
    //                         y.vecdescr,
    //                         CUDA_R_64F,
    //                         CUSPARSE_SPMV_CSR_ALG1,
    //                         externalBuffer);
CHECK_CUSPARSE(    cusparseSpMV(   handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    descr,
                    x.vecdescr,
                    &beta,
                    y.vecdescr,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_CSR_ALG1,
                    externalBuffer));
    cudaFree(externalBuffer);
}


void DCSRSystem::MakeBuffer(DVec &x, DVec &y, double alpha, double beta){
    size_t bufferSize;
    CHECK_CUSPARSE(   cusparseSpMV_bufferSize(handle,
                            cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            descr,
                            x.vecdescr,
                            &beta,
                            y.vecdescr,
                            CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG1,
                            &bufferSize
                            ));
    cudaMalloc(&buffer, bufferSize);
}

void DCSRSystem::PreProcess(DVec &x, DVec &y, double alpha, double beta){
    CHECK_CUSPARSE( cusparseSpMV_preprocess( handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    descr,
                    x.vecdescr,
                    &beta,
                    y.vecdescr,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_CSR_ALG1,
                    buffer));   
}

void DCSRSystem::Axplusy(DVec &x, DVec &y, double alpha, double beta){
    CHECK_CUSPARSE(    cusparseSpMV(   handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    descr,
                    x.vecdescr,
                    &beta,
                    y.vecdescr,
                    CUDA_R_64F,
                    CUSPARSE_SPMV_CSR_ALG1,
                    buffer));
}


void DCSRSystem::ClearBuffer(){
    cudaFree(buffer);
}
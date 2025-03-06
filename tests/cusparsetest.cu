#include "../src/sparse.h"
#include "../src/dvec.h"
#include "../src/dscr_sys.h"
#include "../src/dmatrix.h"

int main(){
    // Data on Host
    Matrix<double> M(4, 4, 0.0);
    for (int i = 0; i < 4;++i)
        M(i, i) = 2;
    printM(M);

    CSR Mcsr(M);
    Mcsr.printCSR();

    Matrix<double> P(2, 2, 1.0), Pout(2,2,0.0),P1(2,2,3.0);
    DMatrix dP(P),dPout(Pout), dP1(P1);
    DVec dVP(dP), dVPout(dPout), dVP1(dP1);

    // cusparse
    DCSRSystem A(M);
    A.MakeBuffer(dVP, dVPout, 1, 0);
    A.Axplusy(dVP, dVPout, 1, 0);
    A.Axplusy(dVP1, dVP, 1, 0);
    cudaDeviceSynchronize();
    A.ClearBuffer();
    dP.deviceMatrixToHost(P);
    printM(P);
    std::cout << std::endl;
    return 0;
}
#include "../src/jacobi.h"
#include "../src/dvec.h"
#include <vector>
#include "../src/matrix.h"

// 5x5 system test
int main()
{
    Matrix<double> A(5, 5, 0);
    A(0, 0) = 4;  A(0, 1) = -1; A(0, 2) = 0;  A(0, 3) = 0;  A(0, 4) = 0;
    A(1, 0) = -1; A(1, 1) = 4;  A(1, 2) = -1; A(1, 3) = 0;  A(1, 4) = 0;
    A(2, 0) = 0;  A(2, 1) = -1; A(2, 2) = 4;  A(2, 3) = -1; A(2, 4) = 0;
    A(3, 0) = 0;  A(3, 1) = 0;  A(3, 2) = -1; A(3, 3) = 4;  A(3, 4) = -1;
    A(4, 0) = 0;  A(4, 1) = 0;  A(4, 2) = 0;  A(4, 3) = -1; A(4, 4) = 3;
    printM(A);
    Jacobi J(makeJacobi(A));
        
    std::vector<double> b = {1, 2, 0, 1, 1};
    std::vector<double> x = {1, 2, 0, 1, 1}; // Initial guess
    DVec d_b(b), d_x(x);

    JacobiSmoother(J, d_x, d_b, 5, 0.001);

    d_x.DVecToHost(x);
    printArray(x.data(), x.size());
    std::cout << std::endl;
    return 0;
}

// 3x3 System
// int main()
// {
//     Matrix<double> A(3,3,0);
//     A(0, 0) = 4;
//     A(0, 1) = 1;
//     A(0, 2) = 1;
//     A(1, 0) = 1;
//     A(1, 1) = 3;
//     A(1, 2) = -1;
//     A(2, 0) = 1;
//     A(2, 1) = -1;
//     A(2, 2) = 2;
//     printM(A);
//     DCSRSystem dA(A);
//     std::vector<double> b = {1, 2, 3};
//     std::vector<double> x = {1, 2, 3};
//     DVec d_b(b),d_x(x);
//     CGSolver(dA, d_x, d_b, 20, 0.0001);
//     d_x.DVecToHost(x);
//     printArray(x.data(), x.size());
//     std::cout << std::endl;
// }
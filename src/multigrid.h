#pragma once
#include "jacobi.h"
#include "dscr_sys.h"

struct MultiGridTwoLevel{
    Jacobi Ah;
    DCSRSystem Prolong;
    DCSRSystem Restrict;
    Jacobi A2h;
    MultiGridTwoLevel(const Matrix<double> &Ah_) :
                     Ah(makeJacobi(Ah_)), A2h(makeJacobi(makeA2h(Ah_,1))), Prolong(makeProlong(Ah_.nr,1)), Restrict(makeProlong(Ah_.nr,1)) {}
};


/// @brief Two Level multigrid solver
/// @param MG 
/// @param x 
/// @param y 
/// @param maxItr 
void MGSolver(const MultiGridTwoLevel &MG, DVec &x, DVec &y, int maxItr);

/// @brief Make Prolongation operator
/// @param Ah_nr (In) : Number of rows in Ah, assuming Ah is symmetric 
/// @param level (In) :
/// @return 
Matrix<double> makeProlong(int Ah_nr, int level);

/// @brief Make Restriction operator
/// @param Ah_nr 
/// @param level 
/// @return 
Matrix<double> makeRestrict(int Ah_nr, int level);

/// @brief Make A2h in Matrix<double> format
/// @param Ah_ 
/// @return 
Matrix<double> makeA2h(const Matrix<double> &Ah, int level);

/// @brief Do C = A*B
/// @param A 
/// @param B 
/// @return 
Matrix<double> MatrixMul(const Matrix<double> &A, const Matrix<double> &B);
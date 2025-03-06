#include "multigrid.h"

void MGSolver(const MultiGridTwoLevel &MG, DVec &x, DVec &y, int maxItr){

    // ToDo5: Assemble multigrid solver
}


Matrix<double> makeProlong(int Ah_nr, int level){

    Matrix<double> Prolong(Ah_nr, static_cast<int>(Ah_nr / (2 * level)), 0);

    // ToDo1: Make Prolongation matrix in dense matrix form (will be converted later)

    return Prolong;
}

Matrix<double> makeRestrict(int Ah_nr, int level){

    Matrix<double> Restrict(static_cast<int>(Ah_nr / (2 * level)), Ah_nr, 0);

    // ToDo2: Make Prolongation matrix in dense matrix form (will be converted later)

    return Restrict;
}

Matrix<double> makeA2h(const Matrix<double> &Ah, int level){
    Matrix<double> Prolong = makeProlong(Ah.nr, level);
    Matrix<double> Restrict = makeRestrict(Ah.nr, level);

    // ToDo4: Make Prolongation matrix in dense matrix form (will be converted later)

    Matrix<double> A2h(5, 5, 0); // dummy placeholder

    return A2h;
}

Matrix<double> MatrixMul(const Matrix<double> &A, const Matrix<double> &B){

    // ToDo3: Make Dense matrix mul operator, needed to make A2h

    Matrix<double> DUMMY(5, 5, 0); // dummy placeholder

    return DUMMY;
}
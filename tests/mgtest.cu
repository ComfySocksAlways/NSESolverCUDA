#include "../src/jacobi.h"
#include "../src/dvec.h"
#include <vector>
#include "../src/matrix.h"
#include "../src/multigrid.h"
#include <iostream>

int main()
{
    Matrix<double> Prolong = makeProlong(6,1);
    printM(Prolong);
}
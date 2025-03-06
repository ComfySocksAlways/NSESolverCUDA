#include "../src/printvtk.h"
#include "../src/matrix.h"
#include "../src/constants.h"

int main(){

    Config config("config.ini");

    std::cout << config.dt << std::endl;
    Matrix<double> u(5, 5, 1), v(5, 5, 1);

    writevtk(u, v, "vtktest.vtk");

    return 0;
}
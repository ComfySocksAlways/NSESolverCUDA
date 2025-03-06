#include "printvtk.h"


void writevtk(const Matrix<double> &u, const Matrix<double> &v, const std::string &filename){

    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // VTK Header
    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "2D velocity field\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET STRUCTURED_GRID\n";
    vtkFile << "DIMENSIONS " << u.nc << " " << u.nr << " 1\n";
    vtkFile << "POINTS " << u.nc * u.nr << " float\n";

    vtkFile << std::fixed << std::setprecision(8);

    // Write Points
    for (int i = 0; i < u.nr; ++i) 
        for (int j = 0; j < u.nc; ++j)
            vtkFile << static_cast<float>(j) << " " << static_cast<float>(i) << " 1.0\n";
    

    // Write Velocity
    vtkFile << "POINT_DATA " << u.nc * u.nr << "\n";
    vtkFile << "VECTORS velocity double\n";
    for (int i = 0; i < u.nr; ++i) 
        for (int j = 0; j < u.nc; ++j)
            vtkFile << u(i, j) << " " << v(i, j) << " 0.0\n";

    vtkFile.close();
}
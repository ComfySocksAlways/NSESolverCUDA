#pragma once
#include "matrix.h"
#include <fstream>
#include <filesystem>
#include <iostream>

void writevtk(const Matrix<double> &u, const Matrix<double> &v, const std::string &filename);
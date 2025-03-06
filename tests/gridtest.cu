#include "../src/grid.h"
#include "../src/matrix.h"
#include<string>
#include <iostream>

int main(int argc, char *argv[]){

    double dx = std::stod(argv[1]);
    double dy = std::stod(argv[2]);
    LidDrivenCavity grid(dx, dy);
    std::cout << grid.nx << " " << grid.ny << std::endl;
}
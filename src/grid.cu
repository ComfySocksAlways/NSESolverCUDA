#include "grid.h"

void LidDrivenCavity::setBoundary(){
    NT.setRow(0, NodeType::GHOST);
    NT.setRow(NT.nr-1, NodeType::GHOST);
    NT.setCol(0, NodeType::GHOST);
    NT.setCol(NT.nc-1, NodeType::GHOST);
    // Change: This is not a wall
    //          wall is assumed to lie b/w nr-1 and nr-2
    //          i.e. the ghost point and it's adjacent
    // u.setRow(u.nr - 2, 1.0);
}
#include "../src/pressure.h"
#include "../src/matrix.h"
#include "../src/dscr_sys.h"
#include "../src/constants.h"
int main()
{
    Config config("config.ini");
    sethConfigPressure(config);
    setdConfigPressure(config);
    Matrix<double> hAp = makePressurePoissonCoeff(6, 6);
    DCSRSystem dcsrAp(hAp);
    checkaccess<<<1, 1>>>();
    return 0;
}
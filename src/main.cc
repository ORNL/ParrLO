#include <iostream>
#include <vector>
#include <stdlib.h>

#include "blas.h"

int main(int argc, char** argv)
{
    // read in an int (dimension)
    const int n = atoi( argv[1] );

    // build vector of values
    std::vector<double> v1(n);
    for (auto& it : v1)
    {
        it = 1.5;
    }

    //blas1 call
    int ione = 1;
    double alpha = 2.;
    C_DSCAL(&n, &alpha, &v1[0], &ione);

    // print out rescaled vector
    for (auto& it : v1)
    {
        std::cout<< it << std::endl;
    }

    return 0;
}


#include "MatrixClasses/Replicated_decl.hpp"

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include <iostream>

int main(int argc, char** argv)
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        return 1;
    }

    magma_int_t magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA INIT SUCCESS" << std::endl;
    }
    else
    {
        if (magmalog == MAGMA_ERR_UNKNOWN)
            std::cout << "MAGMA INIT FAILS UNKNOWN ERROR" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cout << "MAGMA INIT FAILS HOST ALLOC" << std::endl;
        return 1;
    }

    // dimension of matrix
    const int n = 10;

    Replicated A(n, MPI_COMM_WORLD);

    // initialize with random values in interval [-1,1]
    A.initializeRandomSymmetric();

    A.printMatrix();

    double norm = A.maxNorm();

    std::cout << "Max Norm of A = " << norm << std::endl;

    // check if norm consistent with random values in [-1,1]
    if (norm > 1.)
    {
        std::cerr << "Max Norm larger than 1!!!\n" << std::endl;
        return 1;
    }

    if (norm < 0.5)
    {
        std::cerr << "Max Norm smaller than 0.5!!!" << std::endl;
        return 1;
    }

    std::cout << "TEST SUCCESSFUL" << std::endl;

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI_Finalize() failed!!!" << std::endl;
        return 1;
    }

    return 0;
}

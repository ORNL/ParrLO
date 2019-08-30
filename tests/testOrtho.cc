#include "MatrixClasses/Matrix_decl.hpp"
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unistd.h>

#ifndef USE_MAGMA
#include "magma_v2.h"
#endif

int main(int argc, char** argv)
{

    int i = MPI_Init(&argc, &argv);

    if (i != MPI_SUCCESS)
    {
    }
    else
    {

        MPI_Comm lacomm;

        MPI_Comm_dup(MPI_COMM_WORLD, &lacomm);

        MPI_Barrier(lacomm);
        std::cout << "MPI SUCCESS" << i << std::endl;

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);

#ifdef USE_MAGMA
        magma_init();
#else

#endif
        const int nrows = 20;
        const int ncols = 10;

        Matrix A(nrows, ncols, lacomm);
        Matrix B(nrows, ncols, lacomm);
        MPI_Barrier(lacomm);

        A.randomInitialize();
        A.scaleMatrix(0.01);
        A.printMatrix();

        sleep(3);
        MPI_Barrier(lacomm);

        B.identityInitialize();
        B.printMatrix();
        sleep(3);
        MPI_Barrier(lacomm);

        A.matrixSum(B);
        MPI_Barrier(lacomm);
        sleep(3);
        A.printMatrix();
        sleep(3);
        MPI_Barrier(lacomm);

        A.orthogonalize(10, 0.1);

        MPI_Barrier(lacomm);
        sleep(3);

        double dfo        = 0.0;
        const double toll = 5.e-2;

        dfo = A.orthogonalityCheck();

        MPI_Barrier(lacomm);

        if (dfo < toll)
        {
            if (comm_rank == 0)
                std::cout << "Orthogonalized A with dfo" << dfo << std::endl;
            A.printMatrix();
            MPI_Comm_free(&lacomm);

#ifdef USE_MAGMA
            magma_finalize();
#else

#endif

            MPI_Finalize();

            return 0;
        }
        else
        {
            if (comm_rank == 0)
                std::cout << "Orthogonality test failed A with dof" << dfo
                          << std::endl;
            MPI_Comm_free(&lacomm);

#ifdef USE_MAGMA
            magma_finalize();
#else

#endif

            MPI_Finalize();

            return 1;
        }
    }
}

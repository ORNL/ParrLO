#include "MatrixClasses/Matrix_decl.hpp"
#include "MatrixClasses/Timer.hpp"
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unistd.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main(int argc, char** argv)
{

    int i = MPI_Init(&argc, &argv);

    MPI_Comm lacomm;

    MPI_Comm_dup(MPI_COMM_WORLD, &lacomm);

    // std::cout << "MPI SUCCESS" << i << std::endl;

    int comm_rank, comm_size;
    MPI_Comm_rank(lacomm, &comm_rank);
    MPI_Comm_size(lacomm, &comm_size);

    std::string name = "test_rescaling";
    Timer matrix_time(name);
    matrix_time.start();

#ifdef USE_MAGMA
    magma_init();
#else

#endif
    // matrix_time.start();

    int nrows = 2;
    int ncols = 2;
    if (comm_rank == 0)
    {
        std::cout << "Matrix size: " << nrows << "x" << ncols << std::endl;
    }

    Matrix A(nrows, ncols, lacomm);
    Matrix B(nrows, ncols, lacomm);

    //	A.zeroInitialize();
    B.randomInitialize();
    A.identityInitialize();
    A.printMatrix();

    A.scaleMatrix(0.0);
    A.matrixSum(B);
    A.printMatrix();

    double departure_from_orthogonality = 0.0;
    departure_from_orthogonality        = A.orthogonalityCheck();

    if (comm_rank == 0)
        std::cout << "Departure from orthogonality before re-orthogonalizing: "
                  << departure_from_orthogonality << std::endl;

    std::string name_ortho = "orthogonalization";
    Timer orthogonalization_timer(name_ortho);

    orthogonalization_timer.start();
    A.orthogonalize_direct_method();
    orthogonalization_timer.stop();

    A.printMatrix();

    if (comm_rank == 0) std::cout << "Orthogonalized A" << std::endl;
    // A.printMatrix();

    if (comm_rank == 0) std::cout << "Orthogonality check A" << std::endl;

    departure_from_orthogonality = A.orthogonalityCheck();

    if (comm_rank == 0)
        std::cout << "Departure from orthogonality after re-orthogonalizing: "
                  << departure_from_orthogonality << std::endl;

    MPI_Comm_free(&lacomm);

    matrix_time.stop();

    matrix_time.print(std::cout);
    orthogonalization_timer.print(std::cout);

    Replicated::printTimers(std::cout);
    Matrix::printTimers(std::cout);

#ifdef USE_MAGMA
    magma_finalize();
#else

#endif

    MPI_Finalize();

    return 0;
}

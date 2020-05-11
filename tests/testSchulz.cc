#include "MatrixClasses/Replicated_decl.hpp"
#include "MatrixClasses/Timer.hpp"

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include <iostream>
#include <mpi.h>
#include <vector>
#ifdef NCCL_COMM
#include "nccl.h"
#endif

int main(int argc, char** argv)
{
    std::cout << "Test Schulz iterative solver" << std::endl;

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

#ifdef NCCL_COMM
    ncclUniqueId id;
    ncclComm_t nccl_world_comm;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclCommInitRank(&nccl_world_comm, size, id, rank);
#else
    int nccl_world_comm = 0;
#endif

    std::string name = "test_Schultz";
    Timer tstime(name);
    tstime.start();

    // dimension of matrix
    const int n = 10;

    Replicated A(n, MPI_COMM_WORLD, nccl_world_comm);

    // initialize with random values in interval [-0.1,0.1]
    A.initializeRandomSymmetric();
    A.scale(0.1);

    Replicated B(n, MPI_COMM_WORLD, nccl_world_comm);

    B.setDiagonal(1.);
    B.add(0.1, A);
    B.printMatrix();

    // C is a copy of B
    Replicated C(B);

    C.InvSqrt();
    const double normC = C.maxNorm();
    if (std::isnan(normC))
    {
        std::cout << "Max Norm of C is NaN!!!" << std::endl;
        return 1;
    }

    int count_iter = B.SchulzCoupled(20, 1.e-6, "relative", 1);

    B.add(-1., C);

    const double normdiff = B.maxNorm();
    std::cout << "Norm Difference: " << normdiff << std::endl;
    std::cout << "Iterations for Schulz iteration to converge: " << count_iter
              << std::endl;
    if (std::isnan(normdiff))
    {
        std::cout << "Difference is NaN!!!" << std::endl;
        return 1;
    }

    const double tol = 1.e-6;
    std::cout << "Difference:\n";
    B.printMatrix();

    if (normdiff > tol)
    {
        std::cout << "Difference larger than tol!!!" << std::endl;
        return 1;
    }

    magmalog = magma_finalize();

    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA FINALIZE SUCCESS" << std::endl;
    }
    else
    {
        if (magmalog == MAGMA_ERR_UNKNOWN)
            std::cout << "MAGMA FINALIZE FAILS UNKNOWN ERROR" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cout << "MAGMA FINALIZE FAILS HOST ALLOC" << std::endl;
        return 1;
    }

    std::cout << "TEST SUCCESSFUL" << std::endl;

    tstime.stop();
    tstime.print(std::cout);

    // Print timers for operations performed on Replicated matrix
    B.printTimers(std::cout);

#ifdef NCCL_COMM
    ncclCommDestroy(nccl_world_comm);
#endif

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI_Finalize() failed!!!" << std::endl;
        return 1;
    }

    return 0;
}

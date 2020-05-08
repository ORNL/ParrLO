#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include "../src/MatrixClasses/Timer.hpp"
#include "mpi.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << " Requires two input argument!" << std::endl;
        return 1;
    }

    const int N     = atoi(argv[1]);
    const int nreps = atoi(argv[2]);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Timer time_test("test");
    time_test.start();

    std::cout << "N = " << N << std::endl;

    if (N % 2 != 0)
    {
        std::cout << "N has to be even for this test!" << std::endl;
        return 1;
    }
    const size_t N2 = N * N;

    Timer time_init("init");
    time_init.start();

#ifdef USE_MAGMA
    magma_int_t magmalog = magma_init();
    if (magmalog != MAGMA_SUCCESS)
    {
        std::cout << "MAGMA INIT FAILED" << std::endl;
        return 1;
    }

    // if N is even
    const size_t Nd = N / 2;

    magma_device_t device;
    magma_queue_t queue;

    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    MPI_Request reqs[2];

    magma_int_t ld = magma_roundup(N, 32);

    std::vector<double> A(N2, 2.0);
    std::vector<double> B(N2, 3.0);
    std::vector<double> C(N2, 0.0);

    // std::cout<<"print A host"<<std::endl;
    // magma_dprint(N,N,A.data(),N);
    // std::cout<<"print B host"<<std::endl;
    // magma_dprint(N,N,B.data(),N);

    double* dA;
    double* dB;
    double* dC;

    magma_int_t retA = magma_dmalloc(&dA, ld * N);
    magma_int_t retB = magma_dmalloc(&dB, ld * N);
    magma_int_t retC = magma_dmalloc(&dC, ld * N);

    magma_dsetmatrix(N, N, A.data(), N, dA, ld, queue);
    magma_dsetmatrix(N, N, B.data(), N, dB, ld, queue);

    time_init.stop();

    // std::cout<<"print A device rank="<<rank<<std::endl;
    // magma_dprint_gpu(N,Nd,dA,ld,queue);

    // std::cout<<"print B device rank="<<rank<<std::endl;
    // magma_dprint_gpu(N,Nd,dB,ld,queue);

    int j       = (rank + 1) % 2;
    int offset0 = (1 - j) * Nd * ld;
    int offset1 = j * Nd * ld;

    // warmup
    magma_dsetmatrix(N, N, C.data(), N, dC, ld, queue);
    for (int i = 0; i < 10; i++)
    {
        MPI_Irecv(dC + offset1, Nd * ld, MPI_DOUBLE, j, 2345 + j,
            MPI_COMM_WORLD, &reqs[j]);
        MPI_Issend(dC + offset0, Nd * ld, MPI_DOUBLE, j, 2345 + rank,
            MPI_COMM_WORLD, &reqs[rank]);
        MPI_Waitall(2, reqs, MPI_STATUS_IGNORE);
    }

    Timer time_comps("compute single");

    for (int i = 0; i < nreps; i++)
    {
        time_comps.start();

        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, N, N, N, 1.0, dA, ld, dB,
            ld, 0.0, dC, ld, queue);
    }
    magma_queue_sync(queue);

    time_comps.stop();

    // std::cout<<"print C device single"<<std::endl;
    // magma_dprint_gpu(N,N,dC,ld,queue);

    magma_dsetmatrix(N, N, C.data(), N, dC, ld, queue);

    Timer time_compp("compute parallel");
    Timer time_comm("comm");

    for (int i = 0; i < nreps; i++)
    {
        time_compp.start();

        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, N, Nd, N, 1.0, dA, ld,
            dB + offset0, ld, 0.0, dC + offset0, ld, queue);

        magma_queue_sync(queue);

        time_compp.stop();

        time_comm.start();

        // std::cout<<"print C device rank="<<rank<<std::endl;
        // magma_dprint_gpu(N,Nd,dC+offset0,ld,queue);

        MPI_Irecv(dC + offset1, Nd * ld, MPI_DOUBLE, j, 2345 + j,
            MPI_COMM_WORLD, &reqs[j]);
        MPI_Issend(dC + offset0, Nd * ld, MPI_DOUBLE, j, 2345 + rank,
            MPI_COMM_WORLD, &reqs[rank]);

        MPI_Waitall(2, reqs, MPI_STATUS_IGNORE);

        time_comm.stop();
    }
    // std::cout<<"print C device paralell rank="<<rank<<std::endl;
    // magma_dprint_gpu(N,N,dC,ld,queue);

    magma_int_t ret_dA_free = magma_free(dA);
    magma_int_t ret_dB_free = magma_free(dB);
    magma_int_t ret_dC_free = magma_free(dC);

    magma_queue_destroy(queue);

    magmalog = magma_finalize();

    if (magmalog != MAGMA_SUCCESS)
    {
        return 1;
    }

#endif

    time_test.stop();

    time_test.print(std::cout);
    time_comps.print(std::cout);
    time_compp.print(std::cout);
    time_comm.print(std::cout);
    MPI_Finalize();

    return 0;
}

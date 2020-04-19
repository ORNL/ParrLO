#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include "../../src/MatrixClasses/Timer.hpp"
#include "mpi.h"
#include "nccl.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string name3 = "mpi_through_nccl";

    Timer time_mpinccl(name3);

    ncclUniqueId id;
    ncclComm_t nccl_world_comm;
    cudaStream_t s;

    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStreamCreate(&s);

    ncclCommInitRank(&nccl_world_comm, size, id, rank);

    constexpr size_t N  = 3000;
    constexpr size_t N2 = N * N;

    std::vector<double> v1(N2);
    for (auto& it : v1)
    {
        it = 1.0 * rank;
    }

    std::vector<double> vsum_mpinccl(N2, 0.0);

#ifdef USE_MAGMA
    magma_int_t magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        if (rank == 0)
        {
            std::cout << "Number of MPI ranks: " << size << std::endl;
            std::cout << "MAGMA INIT SUCCESS" << std::endl;
        }
    }
    else
    {
        return 1;
    }

    magma_int_t ld = magma_roundup(N, 32);
    magma_device_t device;

    magma_queue_t queue_nccl;

    // magma_int_t cuda_arch = magma_getdevice_arch();

    magma_getdevice(&device);

    magma_queue_create(device, &queue_nccl);

    double* dv1_nccl;
    double* dvsum_nccl;

    magma_int_t ret1           = magma_dmalloc(&dv1_nccl, ld * N);
    magma_int_t retsum_mpinccl = magma_dmalloc(&dvsum_nccl, ld * N);

    // deepcopy from host to device
    magma_dsetmatrix(N, N, v1.data(), N, dv1_nccl, ld, queue_nccl);

    double alpha = 2.0;

    magma_dscal(ld * N, alpha, dv1_nccl, 1, queue_nccl);

    for (int i = 0; i < 100; i++)
    {
        time_mpinccl.start();
        // mpi Allreduce with device nccl
        ncclAllReduce(dv1_nccl, dvsum_nccl, N * ld, ncclDouble, ncclSum,
            nccl_world_comm, s);
        cudaStreamSynchronize(s);
        time_mpinccl.stop();
    }
    // magma_dgetmatrix(N,N,dvsum_nccl,ld,vsum_mpinccl.data(),N,queue_nccl);

    magma_queue_destroy(queue_nccl);

    magma_int_t ret_dv1_nccl_free = magma_free(dv1_nccl);

    magma_int_t ret_dvsum_nccl_free = magma_free(dvsum_nccl);

    magmalog = magma_finalize();

    if (magmalog == MAGMA_SUCCESS)
    {
        if (rank == 0)
        {
            std::cout << "MAGMA FINALIZE SUCCESS" << std::endl;
        }
    }
    else
    {
        return 1;
    }

#endif

    time_mpinccl.print(std::cout);

    cudaStreamDestroy(s);

    ncclCommDestroy(nccl_world_comm);

    MPI_Finalize();

    return 0;
}

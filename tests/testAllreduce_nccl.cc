#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

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

    ncclUniqueId id;
    ncclComm_t nccl_world_comm;
    cudaStream_t s;

    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaStreamCreate(&s);

    ncclCommInitRank(&nccl_world_comm, size, id, rank);

    constexpr size_t N  = 10;
    constexpr size_t N2 = N * N;

    constexpr double alpha = 2.0;

    std::vector<double> v1(N2);
    for (auto& it : v1)
    {
        it = 1.0 * rank;
    }

    std::vector<double> vsum(N2, 0.0);

#ifdef USE_MAGMA
    magma_int_t magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA INIT SUCCESS" << std::endl;
    }
    else
    {
        return 1;
    }

    magma_int_t ld = magma_roundup(N, 32);
    magma_device_t device;
    magma_queue_t queue;

    magma_int_t cuda_arch = magma_getdevice_arch();

    magma_getdevice(&device);

    magma_queue_create(device, &queue);

    double* dv1;
    double* dvsum;
    magma_int_t ret1   = magma_dmalloc(&dv1, ld * N);
    magma_int_t retsum = magma_dmalloc(&dvsum, ld * N);

    // deepcopy from host to device
    magma_dsetmatrix(N, N, v1.data(), N, dv1, ld, queue);

    // calculation on device
    magma_dscal(N * ld, alpha, dv1, 1, queue);

    // mpi Allreduce with device
    // ncclAllReduce(dv1, dvsum, N * ld, ncclDouble, ncclSum, nccl_world_comm,
    // s);
    ncclAllReduce(dv1, dv1, N * ld, ncclDouble, ncclSum, nccl_world_comm, s);

    cudaStreamSynchronize(s);

    // deepcopy from device to host
    // magma_dgetmatrix(N, N, dvsum, ld, vsum.data(), N, queue);
    magma_dgetmatrix(N, N, dv1, ld, v1.data(), N, queue);

    // print from host
    // magma_dprint(N,N,v1.data(),N);
    // magma_dprint(N,N,vsum.data(),N);

    int check;
    check = (int)alpha * (size - 1) * size / 2;

    for (double result : v1)
    {
        if ((int)result != check) return 1;
    }

    magma_queue_destroy(queue);

    magma_int_t ret_dv1_free = magma_free(dv1);

    magma_int_t ret_dvsum_free = magma_free(dvsum);

    magmalog = magma_finalize();

    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA FINALIZE SUCCESS" << std::endl;
    }
    else
    {
        return 1;
    }

#endif
    cudaStreamDestroy(s);

    ncclCommDestroy(nccl_world_comm);

    MPI_Finalize();

    return 0;
}

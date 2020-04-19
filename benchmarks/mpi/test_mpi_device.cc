#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include "../../src/MatrixClasses/Timer.hpp"
#include "mpi.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string name2 = "mpi_through_device";

    Timer time_mpidevice(name2);

    constexpr size_t N  = 3000;
    constexpr size_t N2 = N * N;

    std::vector<double> v1(N2);
    for (auto& it : v1)
    {
        it = 1.0 * rank;
    }

    std::vector<double> vsum_mpidevice(N2, 0.0);

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
    magma_queue_t queue;

    magma_int_t cuda_arch = magma_getdevice_arch();

    magma_getdevice(&device);

    magma_queue_create(device, &queue);

    double* dv1;
    double* dvsum;

    magma_int_t ret1             = magma_dmalloc(&dv1, ld * N);
    magma_int_t retsum_mpidevice = magma_dmalloc(&dvsum, ld * N);

    // deepcopy from host to device
    magma_dsetmatrix(N, N, v1.data(), N, dv1, ld, queue);

    double alpha = 2.0;

    magma_dscal(ld * N, alpha, dv1, 1, queue);

    for (int i = 0; i < 100; i++)
    {
        time_mpidevice.start();
        // mpi Allreduce with device
        MPI_Allreduce(dv1, dvsum, N * ld, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        magma_queue_sync(queue);
        time_mpidevice.stop();
    }

    // deepcopy from device to host
    magma_dgetmatrix(N, N, dvsum, ld, vsum_mpidevice.data(), N, queue);

    magma_queue_destroy(queue);

    magma_int_t ret_dv1_free = magma_free(dv1);

    magma_int_t ret_dvsum_free = magma_free(dvsum);

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

    time_mpidevice.print(std::cout);

    MPI_Finalize();

    return 0;
}

#include "warmup.h"
#include <iostream>

#include "cuda_runtime.h"

void warmup_MPI_pt2pt(MPI_Comm comm)
{
    MPI_Request reqs[2];

    int comm_rank = -1;
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    if (comm_rank == 0) std::cout << "Warmup MPI..." << std::endl;

    const int n = 10000;

    double* send_buffer;
    double* recv_buffer;
#ifdef NCCL_COMM
    cudaMalloc(&send_buffer, n * sizeof(double));
    cudaMalloc(&recv_buffer, n * sizeof(double));
#else
    send_buffer = new double[n];
    recv_buffer = new double[n];
#endif

    for (int j = 0; j < 10; j++)
        for (int i = 0; i < comm_size - 1; i++)
        {
            int dst = (comm_rank + i + 1) % comm_size;
            int src = (comm_size + comm_rank - i - 1) % comm_size;

            MPI_Irecv(send_buffer, n, MPI_DOUBLE, src, i, comm, &reqs[0]);
            MPI_Issend(recv_buffer, n, MPI_DOUBLE, dst, i, comm, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUS_IGNORE);
        }

#ifdef NCCL_COMM
    cudaFree(send_buffer);
    cudaFree(recv_buffer);
#else
    delete[] send_buffer;
    delete[] recv_buffer;
#endif
}

#ifdef NCCL_COMM

void warmup_NCCL(ncclComm_t nccl_comm)
{
    cudaStream_t s;
    cudaStreamCreate(&s);
    double* dwork;
    int datasize = 32 * 100;
    cudaMalloc(&dwork, datasize * sizeof(double));
    for (int i = 0; i < 10; i++)
    {
        ncclAllReduce(
            dwork, dwork, datasize, ncclDouble, ncclSum, nccl_comm, s);
    }
    cudaFree(dwork);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
}

#endif

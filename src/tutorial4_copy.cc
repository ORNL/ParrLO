// tutorial4_copy.cc
#include <cstdlib>
#include <exception>

#include "magma_v2.h"

// -----------------------------------------------------------------------------
// generates random n-by-n A and n-by-nrhs X in CPU host memory
void generate_problem(int n, int nrhs, double* A, int lda, double* X, int ldx)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            A[i + j * lda] = rand() / double(RAND_MAX);
        }
    }
    for (int j = 0; j < nrhs; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            X[i + j * ldx] = rand() / double(RAND_MAX);
        }
    }
}

// -----------------------------------------------------------------------------
// GPU interface: input & output matrices dA and dX in GPU device memory.
// Copy A, X to dA, dX.
int main(int argc, char** argv)
{
    // ... setup A, X in CPU memory;
    // dA, dX in GPU device memory
    magma_init();

    int n = 100, nrhs = 10;
    int lda = n, ldx = n;
    double* A = new double[lda * n];
    double* X = new double[ldx * nrhs];
    int* ipiv = new int[n];

    int ldda = magma_roundup(lda, 32);
    int lddx = magma_roundup(ldx, 32);
    double *dA, *dX;
    magma_dmalloc(&dA, ldda * n);
    magma_dmalloc(&dX, lddx * nrhs);
    assert(dA != nullptr);
    assert(dX != nullptr);

    generate_problem(n, nrhs, A, lda, X, ldx);

    int device;
    magma_queue_t queue;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    // copy A, X to dA, dX
    magma_dsetmatrix(n, n, A, lda, dA, ldda, queue);
    magma_dsetmatrix(n, nrhs, X, ldx, dX, lddx, queue);

    // ... solve AX = B
    int info;
    magma_dgesv_gpu(n, nrhs, dA, ldda, ipiv, dX, lddx, &info);
    if (info != 0)
    {
        throw std::exception();
    }

    // copy result dX to X
    magma_dgetmatrix(n, nrhs, dX, lddx, X, ldx, queue);

    // ... use result in X

    magma_queue_destroy(queue);

    // ... cleanup
    magma_free(dA);
    magma_free(dX);
    delete[] A;
    delete[] X;
    delete[] ipiv;

    magma_finalize();
}

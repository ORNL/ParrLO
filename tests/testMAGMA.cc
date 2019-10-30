#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv)
{
    // dimension of matrix
    const int n  = 10;
    const int n2 = n * n;

    double alpha = 2.;

    // build vector of values on CPU
    std::vector<double> v1(n2);
    for (auto& it : v1)
    {
        it = 1.5;
    }

#ifdef USE_MAGMA

    magma_int_t magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
        std::cout << "MAGMA INIT SUCCESS" << std::endl;
    }
    else
    {
        if (magmalog == MAGMA_ERR_UNKNOWN)
            std::cerr << "MAGMA INIT FAILS UNKNOWN ERROR" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cerr << "MAGMA INIT FAILS HOST ALLOC" << std::endl;
        return 1;
    }

    // alocate data on GPU
    magma_int_t ld = magma_roundup(n, 32);
    magma_device_t device;
    magma_queue_t queue;

    magma_int_t cuda_arch = magma_getdevice_arch();
    std::cout << "Cuda Device Architecture" << cuda_arch << std::endl;

    magma_getdevice(&device);

    magma_queue_create(device, &queue);

    double* dv1;
    magma_int_t ret = magma_dmalloc(&dv1, ld * n);
    if (ret != MAGMA_SUCCESS)
    {
        std::cerr << "magma_dmalloc FAILED!!!" << std::endl;
        return 1;
    }
    // set data on GPU
    magma_dsetmatrix(n, n, &v1[0], n, dv1, ld, queue);

    // rescale data on GPU
    magma_dscal(n * ld, alpha, dv1, 1, queue);

    // get result copy on CPU
    magma_dgetmatrix(n, n, dv1, ld, &v1[0], n, queue);

    magma_dprint(n, n, &v1[0], n);

    const double tol = 1.e-12;
    for (auto ii : v1)
    {
        if (std::abs(v1[ii] - 3.) > tol)
        {
            std::cout << "TEST FAILED" << std::endl;
            return 1;
        }
    }

    magma_queue_destroy(queue);

    magma_int_t ret_free = magma_free(dv1);
    if (ret_free != MAGMA_SUCCESS)
    {
        std::cerr << "magma_free FAILED!!!" << std::endl;
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
            std::cerr << "MAGMA FINALIZE FAILS UNKNOWN ERROR" << std::endl;
        if (magmalog == MAGMA_ERR_HOST_ALLOC)
            std::cerr << "MAGMA FINALIZE FAILS HOST ALLOC" << std::endl;
        return 1;
    }

    std::cout << "TEST SUCCESSFUL" << std::endl;

#endif

    return 0;
}

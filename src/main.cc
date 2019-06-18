#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "blas.h"

int main(int argc, char** argv)
{
    // read in an int (dimension of matrix)
    const int n = atoi( argv[1] );
    const int n2 = n*n;

    double alpha = 2.;

    // build vector of values on CPU
    std::vector<double> v1(n2);
    for (auto& it : v1)
    {
        it = 1.5;
    }

#ifdef USE_MAGMA
    std::cout<<"Run on GPU..."<<std::endl;

    // alocate data on GPU
    int ld =magma_roundup(n, 32);
    int device;
    magma_getdevice(&device);
    magma_queue_t queue;
    magma_queue_create(device, &queue);
    double* dv1;
    magma_int_t ret = magma_dmalloc(&dv1, ld*n);
    assert(ret == MAGMA_SUCCESS);

    // set data on GPU
    magma_dsetmatrix(n,n,&v1[0], n,dv1,ld,queue);

    // rescale data on GPU
    magma_dscal(n*ld, alpha, dv1, 1, queue);

    // get result copy on CPU
    magma_dgetmatrix(n,n,dv1,ld,&v1[0],n,queue);

    magma_queue_destroy( queue );

    magma_free(dv1);
#else

    //blas1 call
    int ione = 1;
    C_DSCAL(&n2, &alpha, &v1[0], &ione);
#endif

    // print out rescaled vector
    for (auto& it : v1)
    {
        std::cout<< it << std::endl;
    }

    return 0;
}


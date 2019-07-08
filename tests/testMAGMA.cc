#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include <iostream>
#include <vector>
#include <stdlib.h>


int main(int argc, char** argv)
{
    // read in an int (dimension of matrix)
    const int n = 10;
    const int n2 = n*n;

    double alpha = 2.;

    // build vector of values on CPU
    std::vector<double> v1(n2);
    for (auto& it : v1)
    {
        it = 1.5;
    }

#ifdef USE_MAGMA
   
    magma_int_t magmalog;   

    magmalog = magma_init();
    if (magmalog == MAGMA_SUCCESS)
    {
     std::cout<<"MAGMA INIT SUCCESS"<<std::endl;
    }else{
     if (magmalog == MAGMA_ERR_UNKNOWN) 
        std::cout<<"MAGMA INIT FAILS UNKNOWN ERROR"<<std::endl;
     if (magmalog == MAGMA_ERR_HOST_ALLOC)
        std::cout<<"MAGMA INIT FAILS HOST ALLOC"<<std::endl;
      return 1;
    } 
    
    // alocate data on GPU
    double* dv1;
    
    magma_int_t ld =magma_roundup(n, 32);
    magma_device_t device;
    magma_queue_t queue;
    magma_int_t cuda_arch;

    cuda_arch = magma_getdevice_arch(); 
    std::cout<<"Cuda Device Architecture"<<cuda_arch<<std::endl; 
  
    magma_getdevice(&device);
    
    magma_queue_create(device,&queue);
    //MAGMA error check  
    magma_dmalloc(&dv1,ld*n);
    //assert(ret == MAGMA_SUCCESS);
    // set data on GPU
    magma_dsetmatrix(n,n,&v1[0],n,dv1,ld,queue);

    // rescale data on GPU
    magma_dscal(n*ld,alpha,dv1,1,queue);

    // get result copy on CPU
    magma_dgetmatrix(n,n,dv1,ld,&v1[0],n,queue);

    magma_dprint(n,n,&v1[0],n);    

    for (auto ii : v1)
    { 
     if(v1[ii]==3)
     { 
      std::cout<<"COMPUTATION SUCCESS"<<std::endl;  
      return 0;
     }else{   
      std::cout<<"COMPUTATION FAIL"<<std::endl; 
      return 1;
     } 
    }
 
    magma_queue_destroy(queue);

    magma_free(dv1);

    magmalog = magma_finalize();

    if (magmalog == MAGMA_SUCCESS)
    {
     std::cout<<"MAGMA FINALIZE SUCCESS"<<std::endl;
    }else{
     if (magmalog == MAGMA_ERR_UNKNOWN)
        std::cout<<"MAGMA FINALIZE FAILS UNKNOWN ERROR"<<std::endl;
     if (magmalog == MAGMA_ERR_HOST_ALLOC)
        std::cout<<"MAGMA FINALIZE FAILS HOST ALLOC"<<std::endl;
       return 1; 
    }

#endif

    return 0;
}


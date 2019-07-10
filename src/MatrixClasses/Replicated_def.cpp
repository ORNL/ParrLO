#include <random>
#include "Replicated_decl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
 
Replicated::Replicated(std::vector<double>& aTa, MPI_Comm comm):lacomm_(comm){

	dim_ = std::sqrt(aTa.size());

	data_.resize(dim_*dim_);
	std::copy(aTa.begin(), aTa.end(), data_.begin());
  
} 

bool Replicated::initialized() const
{
	if(data_initialized_)
		assert( data_.size()==dim_*dim_ );

	return data_initialized_;
}


size_t Replicated::getDim() const { return dim_;}

std::vector<double> Replicated::getCopyData() const
{
	std::vector<double> data_copy(data_.size(),0.0); 
	std::copy(data_.begin(), data_.end(), data_copy.begin());

	return data_copy;

}

const double* Replicated::getDataRawPtr() const
{
	assert( data_.data()!=nullptr );

	return data_.data();

}

void Replicated::printMatrix() const
{
	assert(data_initialized_);
        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm_, &comm_rank);
        MPI_Comm_size(lacomm_, &comm_size);

#ifdef USE_MAGMA
	if(comm_rank==0)
		std::cout<<"MAGMA version of print"<<std::endl;

	std::cout<<"MPI process: "<<comm_rank<<" of "<<comm_size<<std::endl;
	magma_dprint(dim_, dim_, this->getDataRawPtr(), dim_);
#else		
	if(comm_rank==0)
		std::cout<<"Basic implementation of print"<<std::endl;

	std::cout<<"MPI process: "<<comm_rank<<" of "<<comm_size<<std::endl;
	for (size_t j = 0; j < dim_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			std::cout<< data_[i+j*n_rows_]<< "\t";
		}
		std::cout<<"\n"<<std::endl;
	}

#endif

} 

double Replicated::computeFrobeniusNorm()
{
	assert(data_.data() != nullptr);

	double frobSum = 0.0;
	double frobNorm = 0.0;

	std::for_each( data_.begin(),data_.end(), [&frobSum](double x){frobSum += x*x;});	
	frobNorm = std::sqrt(frobSum);
}


void Replicated::Schulz(unsigned int max_iter, double tol)
{
	double alpha = 1.0;
	double beta = 0.0;
	size_t ldc = dim_;

        std::vector<double> hCsum(dim_*dim_, 0.0);

#ifdef USE_MAGMA
	double *dC;

	size_t lddc = magma_roundup(dim_, 32);
	magma_dmalloc( &dC, lddc*dim_ );

        MPI_Allreduce(&data_[0], &hCsum[0], dim_*dim_ , MPI_DOUBLE, MPI_SUM, lacomm_);
	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );
        magma_dsetmatrix( dim_, dim_, &hCsum[0], ldc, dC, lddc, queue );
 
	unsigned int count_iter = 0;
	double relative_residual = 1.0;

	magma_norm_t one_norm = MagmaOneNorm;
	magma_norm_t inf_norm = MagmaInfNorm;
	double *dwork;
	magma_dmalloc( &dwork, lddc );
	double one_norm_value = magmablas_dlange (one_norm, dim_, dim_, dC, lddc, dwork, lddc, queue);
	double inf_norm_value = magmablas_dlange (inf_norm, dim_, dim_, dC, lddc, dwork, lddc, queue);

	//Implementation of Schulz iteration

	double* dI;
	double *dY, *dYaux;
	double *dZ, *dZaux;
	double* dZY;
	double* dIntermediate;
	magma_dmalloc( &dI, lddc*dim_ );
	magma_dmalloc( &dY, lddc*dim_ );
	magma_dmalloc( &dZ, lddc*dim_ );
	magma_dmalloc( &dYaux, lddc*dim_ );
	magma_dmalloc( &dZaux, lddc*dim_ );
	magma_dmalloc( &dZY, lddc*dim_ );
	magma_dmalloc( &dIntermediate, lddc*dim_ );

	magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dI, lddc, queue);
	magma_dcopymatrix(lddc, dim_, dC, lddc, dY, lddc, queue);
	magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dZ, lddc, queue);

	while(count_iter<max_iter & relative_residual>tol)
	{
		//std::cout<<"Iteration count"<<count_iter<<std::endl;
		// Compute ZY
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,dim_,dim_,alpha,dZ,lddc,dY,lddc,beta,dZY,lddc,queue);	
		//magma_dprint_gpu(lddc, dim_, dZY, lddc, queue);

		//Compute 3I-ZY
		magma_dcopymatrix(lddc, dim_, dZY, lddc, dIntermediate, lddc, queue);
		magmablas_dgeadd2(lddc, dim_, 3.0, dI, lddc, -1.0, dIntermediate, lddc, queue);

		//Compute Y(3I-ZY)
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,dim_,dim_,alpha,dY,lddc,dIntermediate,lddc,beta,dYaux,lddc,queue);	

		//Compute (3I-ZY)Z
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,dim_,dim_,alpha,dIntermediate,lddc,dZ,lddc,beta,dZaux,lddc,queue);	

		//Rescale by 1/2
		int val = 0;
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, dim_, dYaux, lddc, queue, &val);
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, dim_, dZaux, lddc, queue, &val);
		magma_dcopymatrix(lddc, dim_, dYaux, lddc, dY, lddc, queue);
		magma_dcopymatrix(lddc, dim_, dZaux, lddc, dZ, lddc, queue);

		count_iter++;
	}

	//Overwrite aTa with the inverse square root
	magma_dgetmatrix( dim_, dim_, dZ, lddc, &data_[0], ldc, queue );

	magma_free(dY);
	magma_free(dZ);
	magma_free(dYaux);
	magma_free(dZaux);
	magma_free(dZY);
	magma_free(dIntermediate);

	magma_free(dC);
	magma_free(dwork);

//	magma_finalize();	
#endif

}



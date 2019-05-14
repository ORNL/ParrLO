#ifndef MATRIX_DEF_HPP
#define MATRIX_DEF_HPP

#include <random>
#include "Matrix_decl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
 
Matrix::Matrix(size_t n, size_t m):n_rows_(n),n_cols_(m){

	assert(n_rows_ >= 0);
	assert(n_cols_ >= 0);
          
        int comm_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        n_rows_local_=floor(n_rows_/comm_size);
        if(comm_rank+1 <= n_rows_ %comm_size)
           n_rows_local_++ ;
         
        global_row_id_.resize(n_rows_local_);
        local_row_id_.resize(n_rows_local_);
  
       //Matrix partitioner
        for (size_t index = 1; index = n_rows_local_; ++index)
           {
           local_row_id_[index-1]  = index -1;
            if(comm_rank+1<= n_rows_%comm_size) 
              global_row_id_[index-1] = (comm_rank+1)*n_rows_local_+index-1;
            else   
              global_row_id_[index-1] = (floor(n_rows_/comm_size) +1)+(n_rows_%comm_size)+floor(n_rows_/comm_size)*(comm_rank-n_rows_%comm_size); 
           }

        data_ = new double[n_rows_local_ * n_cols_];
        //data_.reset( new double[n_rows_ * n_cols_] );


} 

bool Matrix::Initialized() const
{
	return data_initialized_;
}

void Matrix::operator=(const Matrix& B)
{
	assert(B.Initialized());
	assert(B.getNumRows()>=0);
	assert(B.getNumCols()>=0);

	n_rows_ = B.getNumRows();
	n_cols_ = B.getNumCols();
	
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = B(i,j);
		}
	}
	data_initialized_ = true;	
}

double Matrix::operator()(const size_t i, const size_t j) const 
{
// currently does not work with MPI	
        assert(i>=0 & i<n_rows_);
	assert(j>=0 & j<n_cols_);
	
	return data_[i+j*n_rows_];
}


Matrix::Matrix(Matrix& B):n_rows_(B.getNumRows()),n_cols_(B.getNumCols()){

	assert( B.getCopyData()!=NULL );
	assert(n_rows_ >= 0);
	assert(n_cols_ >= 0);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = B(i,j);
		}
	}

	data_initialized_ = true;

} 

void Matrix::zeroInitialize(){

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = 0.0;
		}
	}

	data_initialized_ = true;

}

 
void Matrix::identityInitialize(){

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			if(global_row_id_[i]!=j)
				data_[i+j*n_rows_local_] = 0.0;
			else
				data_[i+j*n_rows_local_] = 1.0;
		}
	}

	data_initialized_ = true;

} 

void Matrix::randomInitialize(){

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1, +1);

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_local_] = dis(gen);
		}
	}

	data_initialized_ = true;

} 

size_t Matrix::getNumRows() const { return n_rows_;}
size_t Matrix::getNumCols() const { return n_cols_;}

double* Matrix::getCopyData() const
{
	assert( data_!=NULL );
	double* data_copy = new double[n_rows_*n_cols_]; 

        for(size_t index = 0; index < n_rows_*n_cols_; ++index)
		data_copy[index] = data_[index];

	return data_copy;

}

void Matrix::printMatrix() const
{
// Currently it does not support MPI
	assert(data_initialized_);

#ifdef USE_MAGMA
	std::cout<<"MAGMA version of print"<<std::endl;
	//magma_dprint(n_rows_, n_cols_, data_.get(), n_rows_);
	magma_dprint(n_rows_, n_cols_, data_, n_rows_);
#else		
	std::cout<<"Basic implementation of print"<<std::endl;
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			std::cout<< data_[i+j*n_rows_]<< "\t";
		}
		std::cout<<"\n"<<std::endl;
	}

#endif

} 

void Matrix::computeFrobeniusNorm()
{

// Currently does not support MPI
	assert(data_ != NULL);
	assert(data_initialized_);

	magma_init();
	double *dA;
	size_t ldda = magma_roundup(n_rows_, 32);
	magma_dmalloc( &dA, ldda*n_cols_ );
	assert( dA != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	//The computation of the infinity norm is not supported yet in MAGMA
	//But we know that ||A||_F <= sqrt( ||A||_1 * ||A||_inf )
	magma_norm_t one_norm = MagmaOneNorm;
	magma_norm_t inf_norm = MagmaInfNorm;
	double *dwork;
	magma_dmalloc( &dwork, ldda );
	double one_norm_value = magmablas_dlange (one_norm, n_rows_, n_cols_, dA, ldda, dwork, ldda, queue);
	double inf_norm_value = magmablas_dlange (inf_norm, n_rows_, n_cols_, dA, ldda, dwork, ldda, queue);
	std::cout<<"Computed One Norm: "<<one_norm_value<<std::endl;
	std::cout<<"Computed Inf Norm: "<<inf_norm_value<<std::endl;
	std::cout<<"Computed Upper Bound for Frobenius Norm: "<<sqrt(one_norm_value * inf_norm_value)<<std::endl;
	magma_finalize();

}


void Matrix::scaleMatrix(double scale_factor)
{
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_local_] = scale_factor * data_[i+j*n_rows_local_];
		}
	}

}

void Matrix::orthogonalize(unsigned int max_iter, double tol)
{
	size_t m = n_cols_;
	size_t n = n_cols_;
	size_t k = n_rows_local_;
	double alpha = 1.0;
	double beta = 0.0;
	double* hC = new double[n_cols_ * n_cols_];
	assert( hC != nullptr );
	double* hA = this->getCopyData();
	double* hB = this->getCopyData();
	size_t lda = n_rows_local_;
	size_t ldb = n_rows_local_;
	size_t ldc = n_cols_;

        double* hCsum  = new double[n_cols_ * n_cols_];

#ifdef USE_MAGMA
	magma_init();

        magma_trans_t transA = MagmaTrans;
	magma_trans_t transB = MagmaNoTrans;
	double *dA, *dB, *dC;

	size_t ldda = magma_roundup(n_rows_local_, 32);
	size_t lddb = magma_roundup(n_rows_local_, 32);
	size_t lddc = magma_roundup(n_cols_, 32);

	magma_dmalloc( &dA, ldda*n_cols_ );
	magma_dmalloc( &dB, lddb*n_cols_ );
	magma_dmalloc( &dC, lddc*n_cols_ );

	assert( dA != nullptr );
	assert( dB != nullptr );
	assert( dC != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	// copy A to dA
	magma_dsetmatrix( n_rows_local_, n_cols_, hA, lda, dA, ldda, queue );
	magma_dsetmatrix( n_rows_local_, n_cols_, hB, ldb, dB, lddb, queue );

	/*std::cout<<"Printing hA:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	std::cout<<"Printing hB:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hB, n_rows_);*/
	/*std::cout<<"Printing dA:"<<std::endl;
	magma_dprint_gpu(ldda, n_cols_, dA, ldda, queue);
	std::cout<<"Printing dB:"<<std::endl;
	magma_dprint_gpu(lddb, n_cols_, dB, lddb, queue);*/
	magmablas_dgemm(transA,transB,m,n,k,alpha,dA,ldda,dB,lddb,beta,dC,lddc,queue);	
	magma_dgetmatrix( n_cols_, n_cols_, dC, lddc, hC, ldc, queue );
        
        // sum hC over all processors
        
        MPI_Allreduce(hC, hCsum, n_cols_*n_cols_ , MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        magma_dsetmatrix( n_cols_, n_cols_, hCsum, ldc, dC, lddc, queue );
 
	unsigned int count_iter = 0;
	double relative_residual = 1.0;

	magma_norm_t one_norm = MagmaOneNorm;
	magma_norm_t inf_norm = MagmaInfNorm;
	double *dwork;
	magma_dmalloc( &dwork, lddc );
	double one_norm_value = magmablas_dlange (one_norm, n_cols_, n_cols_, dC, lddc, dwork, lddc, queue);
	double inf_norm_value = magmablas_dlange (inf_norm, n_cols_, n_cols_, dC, lddc, dwork, lddc, queue);
	std::cout<<"Computed Upper Bound for Frobenius Norm of A^T*A: "<<sqrt(one_norm_value * inf_norm_value)<<std::endl;

	//Implementation of Schulz iteration

	double* dI;
	double *dY, *dYaux;
	double *dZ, *dZaux;
	double* dZY;
	double* dIntermediate;
	magma_dmalloc( &dI, lddc*n_cols_ );
	magma_dmalloc( &dY, lddc*n_cols_ );
	magma_dmalloc( &dZ, lddc*n_cols_ );
	magma_dmalloc( &dYaux, lddc*n_cols_ );
	magma_dmalloc( &dZaux, lddc*n_cols_ );
	magma_dmalloc( &dZY, lddc*n_cols_ );
	magma_dmalloc( &dIntermediate, lddc*n_cols_ );

	magmablas_dlaset(MagmaFull, lddc, n_cols_, 0.0, 1.0, dI, lddc, queue);
	magma_dcopymatrix(lddc, n_cols_, dC, lddc, dY, lddc, queue);
	magmablas_dlaset(MagmaFull, lddc, n_cols_, 0.0, 1.0, dZ, lddc, queue);

	while(count_iter<max_iter & relative_residual>tol)
	{
		std::cout<<"Iteration count"<<count_iter<<std::endl;
		// Compute ZY
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dZ,lddc,dY,lddc,beta,dZY,lddc,queue);	
		magma_dprint_gpu(lddc, n_cols_, dZY, lddc, queue);

		//Compute 3I-ZY
		magma_dcopymatrix(lddc, n_cols_, dZY, lddc, dIntermediate, lddc, queue);
		magmablas_dgeadd2(lddc, n_cols_, 3.0, dI, lddc, -1.0, dIntermediate, lddc, queue);

		//Compute Y(3I-ZY)
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dY,lddc,dIntermediate,lddc,beta,dYaux,lddc,queue);	

		//Compute (3I-ZY)Z
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dIntermediate,lddc,dZ,lddc,beta,dZaux,lddc,queue);	

		//Rescale by 1/2
		int val = 0;
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, n_cols_, dYaux, lddc, queue, &val);
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, n_cols_, dZaux, lddc, queue, &val);
		magma_dcopymatrix(lddc, n_cols_, dYaux, lddc, dY, lddc, queue);
		magma_dcopymatrix(lddc, n_cols_, dZaux, lddc, dZ, lddc, queue);

		count_iter++;
	}

	double* dAortho;
	magma_dmalloc( &dAortho, ldda*n_cols_ );
	magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,ldda,n_cols_,n_cols_,alpha,dA,ldda,dZ,n_cols_,beta,dAortho,ldda,queue);	
	magma_dgetmatrix( n_rows_, n_cols_, dAortho, ldda, data_, lda, queue );

	magma_free(dY);
	magma_free(dZ);
	magma_free(dYaux);
	magma_free(dZaux);
	magma_free(dZY);
	magma_free(dIntermediate);

	magma_free(dA);
	magma_free(dAortho);
	magma_free(dB);
	magma_free(dC);

	magma_finalize();	
#endif
	/*std::cout<<"Printing hC:"<<std::endl;
	magma_dprint(n_cols_, n_cols_, hC, n_cols_);*/

	delete[] hA;
	delete[] hB;
	delete[] hC;	

}

void Matrix::matrix_sum(Matrix& B)
{

	assert(n_rows_ == B.getNumRows());
	assert(n_cols_ == B.getNumCols());

	size_t lda = n_rows_;
	size_t ldb = n_rows_;

	double* hA = this->getCopyData();
	//std::cout<<"Printing hA: "<<std::endl;
	//magma_dprint(n_rows_, n_cols_, hA, n_rows_);
#ifdef USE_MAGMA
        magma_init();

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	size_t ldda = magma_roundup(n_rows_, 32);
	size_t lddb = magma_roundup(n_rows_, 32);

	double *dA;
	magma_dmalloc( &dA, ldda * n_cols_ );
	magma_dsetmatrix( n_rows_, n_cols_, hA, n_rows_, dA, ldda, queue );
	//std::cout<<"Printing dA: "<<std::endl;
	//magma_dprint_gpu(n_rows_, n_cols_, dA, ldda, queue);
	double *dB;
	magma_dmalloc( &dB, lddb * n_cols_ );
	magma_dsetmatrix( n_rows_, n_cols_, B.getCopyData(), n_rows_, dB, lddb, queue );
	//std::cout<<"Printing dB: "<<std::endl;
	//magma_dprint_gpu(n_rows_, n_cols_, dB, lddb, queue);
        magmablas_dgeadd (ldda, n_cols_, 1.0, dB, lddb, dA, ldda, queue);
	magma_dgetmatrix( n_rows_, n_cols_, dA, ldda, data_, n_rows_, queue );
	//std::cout<<"Printing hA: "<<std::endl;
	//magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	magma_free(dA);
	magma_free(dB);

	magma_finalize();
#else
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			data_[i+j*n_rows_] += B(i,j);
		}
	}
#endif
}

#endif

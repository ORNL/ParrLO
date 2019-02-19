#ifndef MATRIX_DEF_HPP
#define MATRIX_DEF_HPP

#include <random>
#include "Matrix_decl.hpp"
#include <iostream>
#include <cassert>
 
Matrix::Matrix(size_t n, size_t m):n_rows_(n),n_cols_(m){

	assert(n_rows_ >= 0);
	assert(n_cols_ >= 0);

        //data_ = new double[n_rows_ * n_cols_];
        data_.reset( new double[n_rows_ * n_cols_] );

} 

void Matrix::ZeroInitialize(){

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_] = 0.0;
		}
	}

	data_initialized_ = true;

} 

void Matrix::RandomInitialize(){

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1, +1);

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_] = dis(gen);
		}
	}

	data_initialized_ = true;

} 

size_t Matrix::getNumRows(){ return n_rows_;}
size_t Matrix::getNumCols(){ return n_cols_;}

double* Matrix::getCopyData() const
{
	assert( data_!=NULL );
	double* data_copy = new double[n_rows_*n_cols_]; 

        for(size_t index = 0; index < n_rows_*n_cols_; ++index)
		data_copy[index] = data_[index];

	return data_copy;

}

void Matrix::printMatrix(){

	assert(data_initialized_);

#ifdef USE_MAGMA
	std::cout<<"MAGMA version of print"<<std::endl;
	magma_dprint(n_rows_, n_cols_, data_.get(), n_rows_);
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

void Matrix::orthogonalize()
{
	magma_init();
        magma_trans_t transA = MagmaTrans;
	magma_trans_t transB = MagmaNoTrans;
	size_t m = n_cols_;
	size_t n = n_cols_;
	size_t k = n_rows_;
	double alpha = 1.0;
	double beta = 0.0;
	double* hC = new double[n_cols_ * n_cols_];
	assert( hC != nullptr );
	double* hA = this->getCopyData();
	double* hB = this->getCopyData();
	size_t lda = n_rows_;
	size_t ldb = n_rows_;
	size_t ldc = n_cols_;
	size_t ldda = n_rows_;
	size_t lddb = n_rows_;
	size_t lddc = n_cols_;

	double *dA, *dB, *dC;
	magma_dmalloc( &dA, n_rows_*n_cols_ );
	magma_dmalloc( &dB, n_rows_*n_cols_ );
	magma_dmalloc( &dC, n_cols_*n_cols_ );
	/*assert( dA != nullptr );
	assert( dB != nullptr );
	assert( dC != nullptr );*/

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	// copy A to dA
	magma_dsetmatrix( n_rows_, n_cols_, hA, lda, dA, ldda, queue );
	magma_dsetmatrix( n_rows_, n_cols_, hB, ldb, dB, lddb, queue );

	std::cout<<"Printing hA:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	std::cout<<"Printing hB:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hB, n_rows_);
	std::cout<<"Printing dA:"<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dA, n_rows_, queue);
	std::cout<<"Printing dB:"<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dB, n_rows_, queue);

	magmablas_dgemm(transA,transB,m,n,k,alpha,dA,ldda,dB,lddb,beta,dC,lddc,queue);		
	magma_dgetmatrix( n_cols_, n_cols_, dC, lddc, hC, ldc, queue );
	std::cout<<"Printing hC:"<<std::endl;
	magma_dprint(n_cols_, n_cols_, hC, n_cols_);
	
	magma_finalize();	
}

void Matrix::matrix_sum()
{
	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	double* hA = this->getCopyData();
	std::cout<<"Printing hA: "<<std::endl;
	magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	double *dA;
	magma_dmalloc( &dA, n_rows_ * n_cols_ );
	magma_dsetmatrix( n_rows_, n_cols_, hA, n_rows_, dA, n_rows_, queue );
	std::cout<<"Printing dA: "<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dA, n_rows_, queue);

}

#endif

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
	double* sts = new double[n_cols_ * n_cols_];

}

#endif

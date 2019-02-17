#ifndef RANDOMMATRIX_DEF_HPP
#define RANDOMMATRIX_DEF_HPP

#include <random>
#include "RandomMatrix_decl.hpp"
#include <iostream>
 
RandomMatrix::RandomMatrix(size_t n, size_t m):n_rows_(n),n_cols_(m){

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1, +1);

        //data_ = new double[n_rows_ * n_cols_];
        data_.reset( new double[n_rows_ * n_cols_] );

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_] = dis(gen);
		}
	}

} 


void RandomMatrix::getMatrix(){

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			std::cout<< data_[i+j*n_rows_]<< "\t";
		}
		std::cout<<"\n"<<std::endl;
	}

} 

#endif

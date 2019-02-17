#ifndef RANDOMMATRIX_DECL_HPP
#define RANDOMMATRIX_DECL_HPP

#include <memory> //needed for unique pointers

class RandomMatrix{	
	private:
		size_t n_rows_;//number of rows
		size_t n_cols_;//number of columns 
		//double* data_; //pointer to basic data structure
		std::unique_ptr<double[]> data_;
        public:
                RandomMatrix(size_t, size_t); //basic constructor
		void getMatrix(); //It is used to visualize the matrix 
                //reorthogonalize();
};
#endif

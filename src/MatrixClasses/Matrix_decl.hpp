#ifndef MATRIX_DECL_HPP
#define MATRIX_DECL_HPP

#include <memory> //needed for unique pointers

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

class Matrix{	

	private:
		size_t n_rows_;//number of rows
		size_t n_cols_;//number of columns 
		//double* data_; //pointer to basic data structure
		std::unique_ptr<double[]> data_; //I want to avoid that the original data gets corrupted

		bool data_initialized_ = false; 

        public:

                //Constructor
                Matrix(size_t, size_t); //basic constructor

                //Set entries of the matrix to zeros 
                void ZeroInitialize();

                //Set entries of the matrix to random values
                void RandomInitialize();

                //Routines to retrieve info about the size of a matrix
		size_t getNumRows();
		size_t getNumCols();

		//Routines to get a copy fo the data
                double* getCopyData() const; //returns the pointer to a copy of the data

                //Visudalization methods
		void printMatrix(); //It is used to visualize the matrix 

		//MAGMA ROUTINES
                void orthogonalize();
};
#endif

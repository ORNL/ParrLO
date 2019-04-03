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
		double* data_; //pointer to basic data structure
		//std::unique_ptr<double[]> data_; //I want to avoid that the original data gets corrupted

		bool data_initialized_ = false; 

        public:

                //Constructor
                Matrix(size_t, size_t); //basic constructor

		//Copy constructor
		Matrix(Matrix&);

		//Return whether a matrix has initialized data or not
		bool Initialized() const;

                //Set entries of the matrix to zeros 
                void zeroInitialize();

                //Initialize a matrix as the identity matrix 
                void identityInitialize();

                //Set entries of the matrix to random values
                void randomInitialize();

                //Routines to retrieve info about the size of a matrix
		size_t getNumRows() const;
		size_t getNumCols() const;
                
		//Overriding of assignment operator 
		void operator=(const Matrix&); 

		//Operator that returns value of a specific entry
		double operator()(const size_t, const size_t) const ; 

		//Routines to get a copy fo the data
                double* getCopyData() const; //returns the pointer to a copy of the data

                //Visudalization methods
		void printMatrix() const; //It is used to visualize the matrix 

		void computeFrobeniusNorm();

                //Scaling
                void scaleMatrix(double);

		//MAGMA ROUTINES
                void orthogonalize(unsigned int, double);
                void matrix_sum(Matrix&);

		//FRIEND methods
		friend Matrix matrix_matrix_multiply( const Matrix&, const Matrix& );
};
#endif

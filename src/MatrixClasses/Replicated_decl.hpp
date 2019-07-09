#ifndef REPLICATED_DECL_HPP
#define REPLICATED_DECL_HPP

#include <memory> //needed for unique pointers
#include <vector>
#include <algorithm>
#include <mpi.h>
#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

class Replicated{	

	private:
		size_t dim_;//dimension of replicated Matrix
                MPI_Comm lacomm;
		std::vector<double> data_; //pointer to basic data structure
		bool data_initialized_ = false; 
  

        public:

		//Copy constructor
		Replicated(std::vector<double>&, MPI_Comm);

		//Return whether a matrix has initialized data or not
		bool initialized() const;

                //Routines to retrieve info about the size of a matrix
		size_t getDim() const;
               
		//Routine that return a vector
		std::vector<double> getCopyData() const; 

                const double* getDataRawPtr() const; //returns the pointer to a copy of the data

                //Visudalization methods
		void printMatrix() const; //It is used to visualize the matrix 

		//Schulz iteration
		void Schulz(unsigned int max_iter, double tol);

		double computeFrobeniusNorm();
		
};
#endif

#ifndef REPLICATED_DECL_HPP
#define REPLICATED_DECL_HPP

#include <memory> //needed for unique pointers
#include <vector>
#include <algorithm>
#include <mpi.h>
#ifdef USE_MAGMA
#include "magma_v2.h"
#endif
		
double relativeDiscrepancy(size_t, size_t, const double*, const double*);

class Replicated{	

	private:
		size_t dim_;//dimension of replicated Matrix
                MPI_Comm lacomm_;
		double* device_data_; //pointer to basic data structure
		bool data_initialized_ = false; 
  

        public:

		//Copy constructor
		Replicated(double*, size_t, MPI_Comm);

		//Return whether a matrix has initialized data or not
		bool initialized() const;

                //Routine to retrieve info about the size of a matrix
		size_t getDim() const;
               
		//returns the pointer to a copy of the data
                const double* getDeviceDataRawPtr() const;

                //Visualization methods
		void printMatrix() const; //It is used to visualize the matrix 

                //Initialize matrix with random values
                //(for testing purposes)
                void initializeRandomSymmetric();

                //rescale values in device_data_
                void scale(const double);

		//Coupled Schulz iteraion
		void SchulzCoupled(unsigned int max_iter, double tol);

		//Stabilized single Schulz iteraion
		void SchulzStabilizedSingle(unsigned int max_iter, double tol);

		//Friend methods
		//Compute convergence criterion for Schulz iteration
		friend double relativeDiscrepancy(size_t, size_t, const double*, const double*);

		
};
#endif

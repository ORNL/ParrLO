#ifndef MATRIX_DECL_HPP
#define MATRIX_DECL_HPP

#include "Replicated_decl.hpp"
#include <algorithm>
#include <memory> //needed for unique pointers
#include <mpi.h>
#include <vector>
#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

class Matrix
{

private:
    size_t n_rows_; // number of rows
    size_t n_cols_; // number of columns
    MPI_Comm lacomm_;
    std::vector<double> host_data_; // vector for data on host
    double* device_data_; // pointer to basic data structure on gpu
    // std::unique_ptr<double[]> host_data_; //I want to avoid that the original
    // data gets corrupted
    double* replicated_S_;
    size_t n_rows_local_;
    std::vector<size_t> global_row_id_;
    std::vector<size_t> local_row_id_;
    bool host_data_initialized_   = false;
    bool device_data_initialized_ = false;

    // Compute local contributions to aTa
    void computeAtA();

public:
    // Constructor
    Matrix(size_t, size_t, MPI_Comm); // basic constructor

    // Copy constructor
    Matrix(Matrix&);

    // Destructor must be explicitly implemented to free memory on gpu
    ~Matrix();

    // Transfer data frok CPU to GPU
    void transferDataCPUtoGPU();

    // Transfer data frok GPU to CPU
    void transferDataGPUtoCPU();

    // Return whether a matrix has initialized data or not
    bool initialized() const;

    // Set entries of the matrix to zeros
    void zeroInitialize();

    // Initialize a matrix as the identity matrix
    void identityInitialize();

    // Set entries of the matrix to random values
    void randomInitialize();

    // Routines to retrieve info about the size of a matrix
    size_t getNumRows() const;
    size_t getNumRowsLocal() const;
    size_t getNumCols() const;

    // Overriding of assignment operator
    void operator=(const Matrix&);

    // Operator that returns value of a specific entry
    double operator()(const size_t, const size_t) const;

    // Routines to get a copy fo the data
    std::vector<double>
    getCopyHostData() const; // returns the vector copy of the data
    const double* getHostDataRawPtr()
        const; // returns the raw pointer to the data on the host
    const double* getDeviceDataRawPtr()
        const; // returns the pointer to the data on the device
    double* getHostDataRawPtrNonConst(); // returns the raw pointer to the data
                                         // on the host
    double* getDeviceDataRawPtrNonConst(); // returns the pointer to the data on
                                           // the device

    // Visudalization methods
    void printMatrix() const; // It is used to visualize the matrix

    double computeFrobeniusNorm();

    // Scaling
    void scaleMatrix(double);

    // Sum of matrices
    void matrixSum(Matrix&);

    // Routine for orthogonalization
    void orthogonalize(unsigned int, double);

    // Routine to check orthogonality
    double orthogonalityCheck();

    // FRIEND methods
    friend double relativeDiscrepancy(
        size_t, size_t, const double*, const double*);
};
#endif

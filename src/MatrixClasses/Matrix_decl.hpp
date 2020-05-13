#ifndef MATRIX_DECL_HPP
#define MATRIX_DECL_HPP

#include "Replicated_decl.hpp"
#include "Timer.hpp"
#include <algorithm>
#include <memory> //needed for unique pointers
#include <mpi.h>
#include <vector>
#ifdef NCCL_COMM
#include "nccl.h"
#endif

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

class Matrix
{

private:
    size_t n_rows_; // number of rows
    size_t n_cols_; // number of columns
    MPI_Comm lacomm_;

#ifndef NCCL_COMM
    typedef int ncclComm_t;
    ncclComm_t nccllacomm_ = 0;
#else
    ncclComm_t nccllacomm_ = NULL;
#endif

    std::vector<double> host_data_; // vector for data on host
    double* device_data_ = nullptr; // pointer to basic data structure on gpu

    // std::unique_ptr<double[]> host_data_; //I want to avoid that the original
    // data gets corrupted
    double* replicated_S_ = nullptr;
    size_t n_rows_local_;
    std::vector<size_t> global_row_id_;
    std::vector<size_t> local_row_id_;

    // Timers
    static Timer compute_aTa_tm_;
    static Timer matrix_matrix_multiply_tm_;
    static Timer allocate_tm_;
    static Timer free_tm_;
    static Timer ortho_tm_;

    // Boolean variables
    bool host_data_initialized_   = false;
    bool device_data_initialized_ = false;

    // Compute local contributions to aTa
    void computeAtA();

public:
    // Constructor
    Matrix(size_t, size_t, MPI_Comm, ncclComm_t);

    // Copy constructor
    Matrix(Matrix&);

    // Destructor must be explicitly implemented to free memory on gpu
    ~Matrix();

    // Transfer data from CPU to GPU
    void transferDataCPUtoGPU();

    // Transfer data from GPU to CPU
    void transferDataGPUtoCPU();

    // Return whether a matrix has initialized data or not
    bool initialized() const;

    // Set entries of the matrix to zeros
    void zeroInitialize();

    // Initialize a matrix as the identity matrix
    void identityInitialize();

    // Set entries of the matrix to random values
    void randomInitialize();

    // Set columns of the matrix to Gaussian functions
    void gaussianColumnsInitialize(
        double standard_deviation, double center_displacement = 0.0);

    // Set columns of the matrix to Gaussian functions
    void hatColumnsInitialize(
        double support_length_ration, double center_displacement = 0.0);

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

    // Routine wrapper for orthogonalization
    int orthogonalize(std::string method, bool diagonal_rescaling = false,
        unsigned int max_iter = 10, double tol = 1e-4,
        std::string implementation = "original", const int ntasks = 1,
        std::string convergence_check    = "relative",
        int frequency_convergcence_check = 1);

    // Routine for orthogonalization using Schulz iteration
    int orthogonalize_iterative_method(std::string, bool, unsigned int, double,
        std::string, const int, std::string, int);

    // Routine for orthogonalization using the diagonalization of a matrix
    void orthogonalize_direct_invsqrt();

    // Routine for orthogonalization using the diagonalization of a matrix
    void orthogonalize_direct_cholesky();

    // Routine to check orthogonality
    double orthogonalityCheck();

    // Print values of timers
    static void printTimers(std::ostream& os)
    {
        compute_aTa_tm_.print(os);
        matrix_matrix_multiply_tm_.print(os);
        allocate_tm_.print(os);
        free_tm_.print(os);
        ortho_tm_.print(os);
    }

    // FRIEND methods
    friend double relativeDiscrepancy(
        size_t, size_t, const double*, const double*);

    friend double absoluteDiscrepancy(
        size_t, size_t, const double*, const double*);
};
#endif

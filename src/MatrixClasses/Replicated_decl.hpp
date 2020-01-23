#ifndef REPLICATED_DECL_HPP
#define REPLICATED_DECL_HPP

#include "Timer.hpp"
#include <algorithm>
#include <memory> //needed for unique pointers
#include <mpi.h>
#include <vector>
#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

double relativeDiscrepancy(size_t, size_t, const double*, const double*);

class Replicated
{

private:
    size_t dim_      = 0; // dimension of replicated Matrix
    MPI_Comm lacomm_ = NULL;
    double* device_data_; // pointer to basic data structure
    std::vector<double> diagonal_;

    // flag to specify is object is responsible for releasing memory
    // associated with device_data_
    bool own_data_;

    bool data_initialized_ = false;

    // Level of verbosity for printouts - default value is 0 which means silent
    int verbosity_ = 0;

    static Timer allreduce_tm_;
    static Timer copy_tm_;
    static Timer memory_initialization_tm_;
    static Timer memory_free_tm_;
    static Timer pre_rescale_tm_;
    static Timer post_rescale_tm_;
    static Timer schulz_iteration_tm_;
    static Timer single_schulz_iteration_tm_;

    // compute eigenvectors and eigenvalues of matrix
    void diagonalize(double* evecs, std::vector<double>& evals);

    // sum up contributions from all MPI tasks
    void consolidate();

public:
    Replicated(const size_t dim, MPI_Comm, int verbosity = 0);

    // Copy constructor
    Replicated(const Replicated& mat);

    // Build matrix with local (partial) contributions to matrix elements
    Replicated(double*, size_t, MPI_Comm, int verbosity = 0);

    ~Replicated();

    // set all values to 0.
    void reset();

    // Return whether a matrix has initialized data or not
    bool initialized() const;

    // Routine to retrieve info about the size of a matrix
    size_t getDim() const;

    // returns the pointer to a copy of the data
    const double* getDeviceDataRawPtr() const;

    // Visualization methods
    void printMatrix() const; // It is used to visualize the matrix

    // compute max norm of matrix
    double maxNorm() const;

    // Initialize matrix with random values
    //(for testing purposes)
    void initializeRandomSymmetric();

    // rescale values in device_data_
    void scale(const double);

    // add a Replicated matrix with scaling factor
    void add(const double, const Replicated&);

    // set diagonal matrix with uniform value alpha
    void setDiagonal(const double alpha);

    // pre-rescaling of the Replicated matrix
    void preRescale();

    // post-rescaling of the Replicated matrix
    void postRescale();

    // Coupled Schulz iteraion
    void SchulzCoupled(unsigned int max_iter, double tol);

    // Stabilized single Schulz iteraion
    void SchulzStabilizedSingle(unsigned int max_iter, double tol);
    void InvSqrt();

    static void printTimers(std::ostream& os)
    {
        allreduce_tm_.print(os);
        copy_tm_.print(os);
        memory_initialization_tm_.print(os);
        memory_free_tm_.print(os);
        pre_rescale_tm_.print(os);
        post_rescale_tm_.print(os);
        schulz_iteration_tm_.print(os);
        single_schulz_iteration_tm_.print(os);
    }

    // Friend methods
    // Compute convergence criterion for Schulz iteration
    friend double relativeDiscrepancy(
        size_t, size_t, const double*, const double*);
};

#endif

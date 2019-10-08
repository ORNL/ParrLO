#include "Matrix_decl.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

Matrix::Matrix(size_t n, size_t m, MPI_Comm comm)
    : n_rows_(n), n_cols_(m), lacomm_(comm)
{

    int comm_rank, comm_size;
    MPI_Comm_rank(lacomm_, &comm_rank);
    MPI_Comm_size(lacomm_, &comm_size);
    n_rows_local_ = floor(n_rows_ / comm_size);
    if (comm_rank + 1 <= n_rows_ % comm_size) n_rows_local_++;
    global_row_id_.resize(n_rows_local_);
    local_row_id_.resize(n_rows_local_);

    // Matrix partitioner
    // We assume that the prtitioning is only needed row-wise: entire rows are
    // owned by a single MPI process
    for (size_t index = 0; index < n_rows_local_; ++index)
    {
        local_row_id_[index] = index;
        if (comm_rank + 1 <= n_rows_ % comm_size)
            global_row_id_[index] = (comm_rank)*n_rows_local_ + index;
        else
        {
            // We need to count the number of MPI processes before this one that
            // have less rows
            int prev_procs_less_rows = (comm_rank - n_rows_ % comm_size);

            // We count all the global rows that are cumulated before this MPI
            // process
            int all_prev_rows = (n_rows_local_ + 1) * (n_rows_ % comm_size)
                                + n_rows_local_ * prev_procs_less_rows;

            // The global index for rows of this MPI process are the local
            // indices shifted by the total number of rows anticipating this MPi
            // process
            global_row_id_[index] = all_prev_rows + index;
        }
    }

    host_data_.resize(n_rows_local_ * n_cols_);
    // host_data_.reset( new double[n_rows_ * n_cols_] );

    // Allocate memory on gpu
    size_t lda  = n_rows_local_;
    size_t ldda = magma_roundup(n_rows_local_, 32);
    magma_dmalloc(&device_data_, ldda * n_cols_);
    assert(device_data_ != nullptr);
}

Matrix::~Matrix()
{

    int ret;
    //  std::cout<<"calling destructor C++"<<std::endl;
#ifdef USE_MAGMA

    if (device_data_ != nullptr)
    {
        // std::cout<<"calling destructor device_data"<<std::endl;
        ret = magma_free(device_data_);
        //  if(ret==MAGMA_SUCCESS){
        //   std::cout<<"magma free device_data success"<<std::endl;
        //   }
        //    else
        //   {
        if (ret == MAGMA_ERR_INVALID_PTR)
        {
            std::cout << "magma free device_data invalid ptr" << std::endl;
        }
    }
    //  }
    if (replicated_S_ != nullptr)
    {
        //  std::cout<<"calling destructor replicated_S"<<std::endl;
        ret = magma_free(replicated_S_);
        // if(ret==MAGMA_SUCCESS){
        //  std::cout<<"magma free replicated_S success"<<std::endl;
        //  }
        //  else
        //  {
        if (ret == MAGMA_ERR_INVALID_PTR)
        {
            std::cout << "magma free replicated_S invalid ptr" << std::endl;
        }
    }
//  }
#endif
}

void Matrix::transferDataCPUtoGPU()
{

    assert(host_data_initialized_);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t lda  = n_rows_local_;
    size_t ldda = magma_roundup(n_rows_local_, 32);

    // copy A to dA
    magma_dsetmatrix(n_rows_local_, n_cols_, this->getHostDataRawPtr(), lda,
        device_data_, ldda, queue);
    if (!device_data_initialized_) device_data_initialized_ = true;
    magma_queue_destroy(queue);
#endif
}

void Matrix::transferDataGPUtoCPU()
{

    assert(device_data_initialized_);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t lda  = n_rows_local_;
    size_t ldda = magma_roundup(n_rows_local_, 32);

    // copy dA to A
    magma_dgetmatrix(
        n_rows_local_, n_cols_, device_data_, ldda, &host_data_[0], lda, queue);
    if (!host_data_initialized_) host_data_initialized_ = true;

#endif
}

bool Matrix::initialized() const
{
    if (host_data_initialized_)
        assert(host_data_.size() == n_rows_local_ * n_cols_);

    return (host_data_initialized_ & device_data_initialized_);
}

void Matrix::operator=(const Matrix& B)
{
    // It performs only a local copy
    assert(B.initialized());

    n_rows_ = B.getNumRows();
    n_cols_ = B.getNumCols();

    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_local_; ++i)
        {
            host_data_[i + j * n_rows_local_] = B(i, j);
        }
    }
    host_data_initialized_ = true;
}

double Matrix::operator()(const size_t i, const size_t j) const
{
    // For now it is only a local access and it assumes that different
    // matrices are partitioned the same way
#ifdef USE_MAGMA
    size_t ldda = magma_roundup(n_rows_local_, 32);
    assert(device_data_initialized_);
    assert(i < n_rows_local_);
    assert(j < n_cols_);

    return device_data_[i + j * ldda];
#else
    assert(host_data_initialized_);
    assert(i < n_rows_local_);
    assert(j < n_cols_);

    return host_data_[i + j * n_rows_local_];
#endif
}

Matrix::Matrix(Matrix& B)
    : n_rows_(B.getNumRows()),
      n_cols_(B.getNumCols()),
      n_rows_local_(B.getNumRowsLocal())
{

    // For now it is only a local access and it assumes that different
    // matrices are partitioned the same way
    assert(B.getHostDataRawPtr() != nullptr);

    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_local_; ++i)
        {
            host_data_[i + j * n_rows_local_] = B(i, j);
        }
    }

    host_data_initialized_ = true;
    this->transferDataCPUtoGPU();
}

void Matrix::zeroInitialize()
{

    assert(!host_data_initialized_);
    assert(host_data_.size() == n_rows_local_ * n_cols_);

    /*for (size_t j = 0; j < n_cols_; ++j) {
            for (size_t i = 0; i < n_rows_local_; ++i) {
                    host_data_[i+j*n_rows_local_] = 0.0;
            }
    }*/

    host_data_.assign(n_rows_local_ * n_cols_, 0);

    host_data_initialized_ = true;
    this->transferDataCPUtoGPU();
}

void Matrix::identityInitialize()
{

    assert(!host_data_initialized_);

    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_local_; ++i)
        {
            if (global_row_id_[i] != j)
                host_data_[i + j * n_rows_local_] = 0.0;
            else
                host_data_[i + j * n_rows_local_] = 1.0;
        }
    }

    host_data_initialized_ = true;
    this->transferDataCPUtoGPU();
}

void Matrix::randomInitialize()
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, +1);

    assert(!host_data_initialized_);

    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_local_; ++i)
        {
            //*(host_data_ + i + j*n_rows_) = dis(gen);
            host_data_[i + j * n_rows_local_] = dis(gen);
        }
    }

    host_data_initialized_ = true;
    this->transferDataCPUtoGPU();
}

size_t Matrix::getNumRows() const { return n_rows_; }
size_t Matrix::getNumRowsLocal() const { return n_rows_local_; }
size_t Matrix::getNumCols() const { return n_cols_; }

std::vector<double> Matrix::getCopyHostData() const
{
    std::vector<double> host_data_copy(host_data_.size(), 0.0);
    std::copy(host_data_.begin(), host_data_.end(), host_data_copy.begin());

    return host_data_copy;
}

const double* Matrix::getHostDataRawPtr() const
{
    assert(host_data_.data() != nullptr);
    return host_data_.data();
}

const double* Matrix::getDeviceDataRawPtr() const
{
    assert(device_data_ != nullptr);
    return device_data_;
}

double* Matrix::getHostDataRawPtrNonConst()
{
    assert(host_data_.data() != nullptr);
    return host_data_.data();
}

double* Matrix::getDeviceDataRawPtrNonConst()
{
    assert(device_data_ != nullptr);
    return device_data_;
}

void Matrix::printMatrix() const
{
    int comm_rank, comm_size;
    MPI_Comm_rank(lacomm_, &comm_rank);
    MPI_Comm_size(lacomm_, &comm_size);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ldda = magma_roundup(n_rows_local_, 32);
    assert(device_data_initialized_);
    if (comm_rank == 0)
        std::cout << "MAGMA version of print" << std::endl << std::flush;

    std::cout << "MPI process: " << comm_rank << " of " << comm_size
              << std::endl
              << std::flush;
    // magma_dprint(n_rows_local_, n_cols_, this->getHostDataRawPtr(),
    // n_rows_local_);
    magma_dprint_gpu(ldda, n_cols_, this->getDeviceDataRawPtr(), ldda, queue);

    magma_queue_destroy(queue);
#else
    assert(host_data_initialized_);
    if (comm_rank == 0)
        std::cout << "Basic implementation of print" << std::endl << std::flush;

    std::cout << "MPI process: " << comm_rank << " of " << comm_size
              << std::endl
              << std::flush;
    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_; ++i)
        {
            //*(host_data_ + i + j*n_rows_) = dis(gen);
            std::cout << host_data_[i + j * n_rows_] << "\t" << std::flush;
        }
        std::cout << "\n" << std::endl << std::flush;
    }

#endif
}

double Matrix::computeFrobeniusNorm()
{
    assert(host_data_.data() != nullptr);

    int comm_rank, comm_size;
    MPI_Comm_rank(lacomm_, &comm_rank);
    MPI_Comm_size(lacomm_, &comm_size);

    double frobSum    = 0.0;
    double frobSumAll = 0.0;
    double frobNorm   = 0.0;

    std::for_each(host_data_.begin(), host_data_.end(),
        [&frobSum](double x) { frobSum += x * x; });
    MPI_Allreduce(&frobSum, &frobSumAll, 1, MPI_DOUBLE, MPI_SUM, lacomm_);
    frobNorm = std::sqrt(frobSumAll);
}

void Matrix::scaleMatrix(double scale_factor)
{
#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ldda = magma_roundup(n_rows_local_, 32);
    magma_dscal(ldda * n_cols_, scale_factor, device_data_, 1, queue);
    this->transferDataGPUtoCPU();
    magma_queue_destroy(queue);
#else
    std::transform(host_data_.begin(), host_data_.end(), host_data_.begin(),
        [scale_factor](double alpha) { return scale_factor * alpha; });
#endif
}

void Matrix::computeAtA()
{

    size_t m     = n_cols_;
    size_t n     = n_cols_;
    size_t k     = n_rows_local_;
    double alpha = 1.0;
    double beta  = 0.0;

#ifdef USE_MAGMA

    assert(device_data_initialized_);

    magma_trans_t transA = MagmaTrans;
    magma_trans_t transB = MagmaNoTrans;

    size_t ldda = magma_roundup(n_rows_local_, 32);
    size_t lddb = magma_roundup(n_rows_local_, 32);
    size_t lddc = magma_roundup(n_cols_, 32);

    magma_dmalloc(&replicated_S_, lddc * n_cols_);

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    // Compute local contribution to A^T * A
    magmablas_dgemm(transA, transB, m, n, k, alpha, device_data_, ldda,
        device_data_, lddb, beta, replicated_S_, lddc, queue);
    magma_queue_destroy(queue);

#endif
}

void Matrix::orthogonalize(unsigned int max_iter, double tol)
{
    // compute local contributions to Gram matrix
    computeAtA();

    size_t m     = n_cols_;
    size_t n     = n_cols_;
    size_t k     = n_rows_local_;
    double alpha = 1.0;
    double beta  = 0.0;
    size_t lda   = n_rows_local_;
    size_t ldc   = n_cols_;

#ifdef USE_MAGMA

    assert(device_data_initialized_);

    size_t ldda = magma_roundup(n_rows_local_, 32);
    size_t lddc = magma_roundup(n_cols_, 32);

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    Replicated AtA(replicated_S_, n_cols_, lacomm_);
    // AtA.SchulzCoupled(max_iter, tol);
    AtA.SchulzStabilizedSingle(max_iter, tol);

    // Restore orthogonality on columns of A
    double* dAortho;
    magma_dmalloc(&dAortho, ldda * n_cols_);
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, ldda, n_cols_, n_cols_, alpha,
        device_data_, ldda, replicated_S_, lddc, beta, dAortho, ldda, queue);

    // Store re-orthogonalized matrix
    magma_dcopymatrix(
        n_rows_local_, n_cols_, dAortho, ldda, device_data_, ldda, queue);
    // Copy dAortho to A on cpu
    // magma_dgetmatrix( n_rows_local_, n_cols_, dAortho, ldda, &host_data_[0],
    // lda, queue );

    // Free gpu memory
    magma_free(dAortho);

    magma_queue_destroy(queue);

#endif
}

double Matrix::orthogonalityCheck()
{

    size_t m           = n_cols_;
    size_t n           = n_cols_;
    size_t k           = n_rows_local_;
    double discrepancy = 1.0;
    double alpha       = 1.0;
    double beta        = 0.0;
    std::vector<double> hC(n_cols_ * n_cols_, 0.0);
    std::vector<double> hCsum(n_cols_ * n_cols_, 0.0);
    assert(&hC[0] != nullptr);
    assert(&hCsum[0] != nullptr);
    size_t lda = n_rows_local_;
    size_t ldb = n_rows_local_;
    size_t ldc = n_cols_;

#ifdef USE_MAGMA

    magma_trans_t transA = MagmaTrans;
    magma_trans_t transB = MagmaNoTrans;
    double *dC, *dI;

    size_t ldda = magma_roundup(n_rows_local_, 32);
    size_t lddb = magma_roundup(n_rows_local_, 32);
    size_t lddc = magma_roundup(n_cols_, 32);

    magma_dmalloc(&dC, lddc * n_cols_);
    magma_dmalloc(&dI, lddc * n_cols_);

    assert(dC != nullptr);

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    magmablas_dgemm(transA, transB, m, n, k, alpha, this->getDeviceDataRawPtr(),
        ldda, this->getDeviceDataRawPtr(), lddb, beta, dC, lddc, queue);
    magma_dgetmatrix(n_cols_, n_cols_, dC, lddc, &hC[0], ldc, queue);

    // sum hC over all processors
    MPI_Allreduce(
        &hC[0], &hCsum[0], n_cols_ * n_cols_, MPI_DOUBLE, MPI_SUM, lacomm_);
    magma_dsetmatrix(n_cols_, n_cols_, &hCsum[0], ldc, dC, lddc, queue);

    // Initialize identity matrix
    magmablas_dlaset(MagmaFull, lddc, n_cols_, 0.0, 1.0, dI, lddc, queue);

    discrepancy = relativeDiscrepancy(lddc, n_cols_, dC, dI);

    magma_free(dC);
    magma_free(dI);

    magma_queue_destroy(queue);
#endif

    return discrepancy;
}

void Matrix::matrixSum(Matrix& B)
{

    assert(n_rows_ == B.getNumRows());
    assert(n_cols_ == B.getNumCols());

    size_t lda = n_rows_local_;
    size_t ldb = n_rows_local_;

    // double* hA = this->getCopyRawPtrData();
    // std::cout<<"Printing hA: "<<std::endl;
    // magma_dprint(n_rows_, n_cols_, hA, n_rows_);
#ifdef USE_MAGMA
    //        magma_init();

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ldda = magma_roundup(n_rows_local_, 32);
    size_t lddb = magma_roundup(n_rows_local_, 32);

    assert(this->initialized());
    assert(B.initialized());
    magmablas_dgeadd(ldda, n_cols_, 1.0, B.getDeviceDataRawPtr(), lddb,
        this->getDeviceDataRawPtrNonConst(), ldda, queue);
    this->transferDataGPUtoCPU();

    magma_queue_destroy(queue);
// if MAGMA not used reimplement access operator
#else
    for (size_t j = 0; j < n_cols_; ++j)
    {
        for (size_t i = 0; i < n_rows_local_; ++i)
        {
            host_data_[i + j * n_rows_local_] += B(i, j);
        }
    }
#endif
}

#include "Replicated_decl.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

double relativeDiscrepancy(size_t n, size_t m, const double* A, const double* B)
{
    double normA = 0.0;
    double normC = 0.0;

#ifdef USE_MAGMA
    assert(A != nullptr);
    assert(B != nullptr);

    size_t lddc = magma_roundup(n, 32);
    double* C;
    magma_dmalloc(&C, lddc * m);

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    unsigned int count_iter  = 0;
    double relative_residual = 1.0;

    magma_norm_t inf_norm = MagmaInfNorm;
    double* dwork;
    magma_dmalloc(&dwork, lddc);

    // Compute norm of A
    normA = magmablas_dlange(inf_norm, n, m, A, lddc, dwork, lddc, queue);

    // Compute C = A-B
    magma_dcopymatrix(lddc, m, B, lddc, C, lddc, queue);
    magmablas_dgeadd2(lddc, m, 1.0, A, lddc, -1.0, C, lddc, queue);

    // Compute norm of C = A-B
    normC = magmablas_dlange(inf_norm, n, m, C, lddc, dwork, lddc, queue);

    magma_free(C);
    magma_free(dwork);
#endif

    return normC / normA;
}

Replicated::Replicated(const size_t dim, MPI_Comm comm)
    : dim_(dim), lacomm_(comm)
{
    data_initialized_ = false;
    size_t ld         = magma_roundup(dim_, 32);

    magma_dmalloc(&device_data_, dim_ * ld);
    own_data_ = true;
}

Replicated::Replicated(double* partial, size_t dim, MPI_Comm comm)
    : lacomm_(comm), dim_(dim), device_data_(partial)
{
    data_initialized_ = true;
    own_data_         = false;

    // data is sum of partial contributions
    consolidate();
}

Replicated::Replicated(const Replicated& mat)
    : dim_(mat.dim_), lacomm_(mat.lacomm_)
{
    size_t ld = magma_roundup(dim_, 32);

    magma_dmalloc(&device_data_, dim_ * ld);
    own_data_ = true;

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    magma_dcopymatrix(ld, dim_, mat.device_data_, ld, device_data_, ld, queue);

    magma_queue_destroy(queue);

    data_initialized_ = true;
}

Replicated::~Replicated()
{
    int ret;
    //    std::cout<<"calling Replicated destructor"<<own_data_<<std::endl;
    if (own_data_)
    {

        //  std::cout<<"free device_data from replicated destructor"<<std::endl;
        ret = magma_free(device_data_);
        //  if(ret==MAGMA_SUCCESS){
        //    std::cout<<"magma free device_data success rep destr"<<std::endl;
        //  }
        //  else
        //  {
        if (ret == MAGMA_ERR_INVALID_PTR)
        {
            std::cout << "magma free device_data invalid ptr rep destr"
                      << std::endl;
        }
    }
    //  }
}

bool Replicated::initialized() const { return data_initialized_; }

size_t Replicated::getDim() const { return dim_; }

const double* Replicated::getDeviceDataRawPtr() const
{
    assert(device_data_ != nullptr);
    return device_data_;
}

void Replicated::printMatrix() const
{
    assert(data_initialized_);
    int comm_rank, comm_size;
    MPI_Comm_rank(lacomm_, &comm_rank);
    MPI_Comm_size(lacomm_, &comm_size);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);
    size_t lddc = magma_roundup(dim_, 32);

    if (comm_rank == 0) std::cout << "MAGMA version of print" << std::endl;

    std::cout << "MPI process: " << comm_rank << " of " << comm_size
              << std::endl;
    magma_dprint_gpu(lddc, dim_, this->getDeviceDataRawPtr(), lddc, queue);
#else
    if (comm_rank == 0)
        std::cout << "Basic implementation of print" << std::endl;

    std::cout << "MPI process: " << comm_rank << " of " << comm_size
              << std::endl;
    for (size_t j = 0; j < dim_; ++j)
    {
        for (size_t i = 0; i < n_rows_; ++i)
        {
            std::cout << data_[i + j * n_rows_] << "\t";
        }
        std::cout << "\n" << std::endl;
    }

#endif
}

void Replicated::scale(const double alpha)
{
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ld = magma_roundup(dim_, 32);

    magma_dscal(dim_ * ld, alpha, device_data_, 1, queue);

    magma_queue_destroy(queue);
}

void Replicated::add(const double alpha, const Replicated& dA)
{
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ld = magma_roundup(dim_, 32);

    magmablas_dgeadd(
        dim_, dim_, alpha, dA.device_data_, ld, device_data_, ld, queue);

    magma_queue_destroy(queue);
}

double Replicated::maxNorm() const
{
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    double* dwork;
    magma_dmalloc(&dwork, dim_);

    size_t ld = magma_roundup(dim_, 32);

    double norm = magmablas_dlange(
        MagmaMaxNorm, dim_, dim_, device_data_, ld, dwork, dim_, queue);

    magma_queue_destroy(queue);

    magma_free(dwork);

    return norm;
}

void Replicated::SchulzCoupled(unsigned int max_iter, double tol)
{
    double alpha       = 1.0;
    double beta        = 0.0;
    double discrepancy = 1.0;
    size_t ldc         = dim_;
    size_t lddc        = magma_roundup(dim_, 32);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    unsigned int count_iter  = 0;
    double relative_residual = 1.0;

    magma_norm_t one_norm = MagmaOneNorm;
    magma_norm_t inf_norm = MagmaInfNorm;
    double* dwork;
    magma_dmalloc(&dwork, lddc);
    double one_norm_value = magmablas_dlange(
        one_norm, dim_, dim_, device_data_, lddc, dwork, lddc, queue);
    double inf_norm_value = magmablas_dlange(
        inf_norm, dim_, dim_, device_data_, lddc, dwork, lddc, queue);

    // Implementation of Schulz iteration

    double* dI;
    double *dY, *dYaux;
    double *dZ, *dZaux;
    double* dZY;
    double* dIntermediate;
    magma_dmalloc(&dI, lddc * dim_);
    magma_dmalloc(&dY, lddc * dim_);
    magma_dmalloc(&dZ, lddc * dim_);
    magma_dmalloc(&dYaux, lddc * dim_);
    magma_dmalloc(&dZaux, lddc * dim_);
    magma_dmalloc(&dZY, lddc * dim_);
    magma_dmalloc(&dIntermediate, lddc * dim_);

    magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dI, lddc, queue);
    magma_dcopymatrix(lddc, dim_, device_data_, lddc, dY, lddc, queue);
    magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dZ, lddc, queue);

    while (count_iter<max_iter & discrepancy> tol)
    {
        // std::cout<<"Iteration count"<<count_iter<<std::endl;
        // Compute ZY
        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, lddc, dim_, dim_, alpha, dZ,
            lddc, dY, lddc, beta, dZY, lddc, queue);
        // magma_dprint_gpu(lddc, dim_, dZY, lddc, queue);

        // Compute 3I-ZY
        magma_dcopymatrix(lddc, dim_, dZY, lddc, dIntermediate, lddc, queue);
        magmablas_dgeadd2(
            lddc, dim_, 3.0, dI, lddc, -1.0, dIntermediate, lddc, queue);

        // Compute Y(3I-ZY)
        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, lddc, dim_, dim_, alpha, dY,
            lddc, dIntermediate, lddc, beta, dYaux, lddc, queue);

        // Compute (3I-ZY)Z
        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, lddc, dim_, dim_, alpha,
            dIntermediate, lddc, dZ, lddc, beta, dZaux, lddc, queue);

        // Rescale by 1/2
        int val = 0;
        magmablas_dlascl(
            MagmaFull, 0, 0, 2.0, 1.0, lddc, dim_, dYaux, lddc, queue, &val);
        magmablas_dlascl(
            MagmaFull, 0, 0, 2.0, 1.0, lddc, dim_, dZaux, lddc, queue, &val);
        magma_dcopymatrix(lddc, dim_, dYaux, lddc, dY, lddc, queue);

        // Compute discrepancy between consecutive updates of dZ for convergence
        // criterion
        discrepancy = relativeDiscrepancy(dim_, dim_, dZ, dZaux);

        magma_dcopymatrix(lddc, dim_, dZaux, lddc, dZ, lddc, queue);

        count_iter++;
    }

    // Overwrite aTa with the inverse square root
    magma_dcopymatrix(lddc, dim_, dZ, lddc, device_data_, lddc, queue);

    magma_free(dY);
    magma_free(dZ);
    magma_free(dYaux);
    magma_free(dZaux);
    magma_free(dZY);
    magma_free(dIntermediate);

    magma_free(dwork);

#endif
}

void Replicated::SchulzStabilizedSingle(unsigned int max_iter, double tol)
{
    double alpha       = 1.0;
    double beta        = 0.0;
    double discrepancy = 1.0;
    size_t ldc         = dim_;
    size_t lddc        = magma_roundup(dim_, 32);

#ifdef USE_MAGMA
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    unsigned int count_iter  = 0;
    double relative_residual = 1.0;

    magma_norm_t one_norm = MagmaOneNorm;
    magma_norm_t inf_norm = MagmaInfNorm;
    double* dwork;
    magma_dmalloc(&dwork, lddc);
    double one_norm_value = magmablas_dlange(
        one_norm, dim_, dim_, device_data_, lddc, dwork, lddc, queue);
    double inf_norm_value = magmablas_dlange(
        inf_norm, dim_, dim_, device_data_, lddc, dwork, lddc, queue);

    // Implementation of Schulz iteration

    double* dI;
    double *dZ, *dY, *dZaux;
    double* dZY;
    double* dIntermediate;
    magma_dmalloc(&dI, lddc * dim_);
    magma_dmalloc(&dY, lddc * dim_);
    magma_dmalloc(&dZ, lddc * dim_);
    magma_dmalloc(&dZaux, lddc * dim_);
    magma_dmalloc(&dZY, lddc * dim_);
    magma_dmalloc(&dIntermediate, lddc * dim_);

    magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dI, lddc, queue);
    magmablas_dlaset(MagmaFull, lddc, dim_, 0.0, 1.0, dZ, lddc, queue);

    while (count_iter<max_iter & discrepancy> tol)
    {
        // Compute Y = A*Z
        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, lddc, dim_, dim_, alpha,
            device_data_, lddc, dZ, lddc, beta, dY, lddc, queue);

        // std::cout<<"Iteration count"<<count_iter<<std::endl;
        // Compute Z^T*Y for stabilization
        magmablas_dgemm(MagmaTrans, MagmaNoTrans, lddc, dim_, dim_, alpha, dZ,
            lddc, dY, lddc, beta, dZY, lddc, queue);
        // magma_dprint_gpu(lddc, dim_, dZY, lddc, queue);

        // Compute 3I-ZY
        magma_dcopymatrix(lddc, dim_, dZY, lddc, dIntermediate, lddc, queue);
        magmablas_dgeadd2(
            lddc, dim_, 3.0, dI, lddc, -1.0, dIntermediate, lddc, queue);

        // Compute (3I-ZY)Z
        magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, lddc, dim_, dim_, alpha,
            dIntermediate, lddc, dZ, lddc, beta, dZaux, lddc, queue);

        // Rescale by 1/2
        int val = 0;
        magmablas_dlascl(
            MagmaFull, 0, 0, 2.0, 1.0, lddc, dim_, dZaux, lddc, queue, &val);

        // Compute discrepancy between consecutive updates of dZ for convergence
        // criterion
        discrepancy = relativeDiscrepancy(dim_, dim_, dZ, dZaux);

        magma_dcopymatrix(lddc, dim_, dZaux, lddc, dZ, lddc, queue);

        count_iter++;
    }

    // Overwrite aTa with the inverse square root
    magma_dcopymatrix(lddc, dim_, dZ, lddc, device_data_, lddc, queue);

    magma_free(dZ);
    magma_free(dY);
    magma_free(dZaux);
    magma_free(dZY);
    magma_free(dIntermediate);

    magma_free(dwork);

#endif
}

void Replicated::initializeRandomSymmetric()
{
    assert(device_data_ != nullptr);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    // initialize random matrix on CPU
    std::vector<double> work(dim_ * dim_);

    for (size_t j = 0; j < dim_; ++j)
    {
        for (size_t i = 0; i <= j; ++i)
        {
            work[i + j * dim_] = dis(gen);
            if (i != j) work[j + i * dim_] = work[i + j * dim_];
        }
    }

    size_t ld = magma_roundup(dim_, 32);
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    // copy work to device_data_
    magma_dsetmatrix(dim_, dim_, work.data(), dim_, device_data_, ld, queue);

    magma_queue_destroy(queue);

    data_initialized_ = true;
}

void Replicated::reset()
{
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ld = magma_roundup(dim_, 32);

    // set diagonal and offdiagonal values to 0.
    magmablas_dlaset(MagmaFull, dim_, dim_, 0., 0., device_data_, ld, queue);

    magma_queue_destroy(queue);

    data_initialized_ = true;
}

void Replicated::setDiagonal(const double alpha)
{
    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ld = magma_roundup(dim_, 32);

    // set offdiag values to 0, diag to alpha
    magmablas_dlaset(MagmaFull, dim_, dim_, 0., alpha, device_data_, ld, queue);

    magma_queue_destroy(queue);

    data_initialized_ = true;
}

void Replicated::diagonalize(double* evecs, std::vector<double>& evals)
{
    int info;

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    size_t ld = magma_roundup(dim_, 32);

    int lwork  = 2 * dim_ + dim_ * magma_get_ssytrd_nb(dim_);
    int liwork = 3 + 5 * dim_;

    std::vector<double> wa(dim_ * dim_);
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);

    // copy matrix into evecs
    magmablas_dlacpy(MagmaFull, dim_, dim_, device_data_, ld, evecs, ld, queue);

    magma_dsyevd_gpu(MagmaVec, MagmaUpper, dim_, evecs, ld, evals.data(),
        wa.data(), dim_, work.data(), lwork, iwork.data(), liwork, &info);

    magma_queue_destroy(queue);
}

void Replicated::InvSqrt()
{
    double* evecs;
    size_t ld = magma_roundup(dim_, 32);

    magma_int_t ret = magma_dmalloc(&evecs, dim_ * ld);
    assert(ret == MAGMA_SUCCESS);

    double* work;
    ret = magma_dmalloc(&work, dim_ * ld);
    assert(ret == MAGMA_SUCCESS);

    std::vector<double> evals(dim_);

    diagonalize(evecs, evals);

    std::transform(evals.begin(), evals.end(), evals.begin(),
        [](double alpha) { return 1. / sqrt(alpha); });

    // set matrix values to 0.
    reset();

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    // set diagonal values to evals
    magma_dsetvector(dim_, evals.data(), 1, device_data_, ld + 1, queue);

    // multiply diagonal matrix right and left by matrix
    // of eigenvectors
    magmablas_dgemm(MagmaNoTrans, MagmaTrans, ld, dim_, dim_, 1., device_data_,
        ld, evecs, ld, 0., work, ld, queue);
    magmablas_dgemm(MagmaNoTrans, MagmaNoTrans, ld, dim_, dim_, 1., evecs, ld,
        work, ld, 0., device_data_, ld, queue);

    magma_free(evecs);
    magma_free(work);

    magma_queue_destroy(queue);
}

void Replicated::consolidate()
{
    std::vector<double> hC(dim_ * dim_, 0.0);
    std::vector<double> hCsum(dim_ * dim_, 0.0);

    size_t ld = magma_roundup(dim_, 32);

    magma_queue_t queue;
    int device;
    magma_getdevice(&device);
    magma_queue_create(device, &queue);

    // copy from GPU to CPU
    magma_dgetmatrix(dim_, dim_, device_data_, ld, &hC[0], dim_, queue);

    // replicated matrix data is sum of all partial matrices
    MPI_Allreduce(&hC[0], &hCsum[0], dim_ * dim_, MPI_DOUBLE, MPI_SUM, lacomm_);

    magma_dsetmatrix(dim_, dim_, &hCsum[0], dim_, device_data_, ld, queue);

    std::cout << "Printing matrix after MPI_Allreduce SUM:" << std::endl;
    magma_dprint_gpu(ld, dim_, device_data_, ld, queue);
}

#ifndef MATRIX_DEF_HPP
#define MATRIX_DEF_HPP

#include <random>
#include "Matrix_decl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
 
Matrix::Matrix(size_t n, size_t m):n_rows_(n),n_cols_(m){

        int comm_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        n_rows_local_=floor(n_rows_/comm_size);
        if(comm_rank+1 <= n_rows_ %comm_size)
           n_rows_local_++ ;
         
        global_row_id_.resize(n_rows_local_);
        local_row_id_.resize(n_rows_local_);
  
       //Matrix partitioner
       //We assume that the prtitioning is only needed row-wise: entire rows are owned by a single MPI process
        for (size_t index = 0; index < n_rows_local_; ++index)
        {
           local_row_id_[index]  = index;
            if(comm_rank+1<= n_rows_%comm_size) 
              global_row_id_[index] = (comm_rank)*n_rows_local_+index;
            else
            {
	      //We need to count the number of MPI processes before this one that have less rows
	      int prev_procs_less_rows = (comm_rank-n_rows_%comm_size);

	      //We count all the global rows that are cumulated before this MPI process
              int all_prev_rows = (n_rows_local_+1)*(n_rows_%comm_size) + n_rows_local_ * prev_procs_less_rows; 

	      //The global index for rows of this MPI process are the local indices shifted by the total number of rows anticipating this MPi process
	      global_row_id_[index] = all_prev_rows + index;
	    }
        }

	data_.resize(n_rows_local_ * n_cols_);
        //data_ = new double[n_rows_local_ * n_cols_];
        //data_.reset( new double[n_rows_ * n_cols_] );


} 

bool Matrix::initialized() const
{
	if(data_initialized_)
		assert( data_.size()==n_rows_local_*n_cols_ );

	return data_initialized_;
}

void Matrix::operator=(const Matrix& B)
{
	//It performs only a local copy
	assert(B.initialized());

	n_rows_ = B.getNumRows();
	n_cols_ = B.getNumCols();
	
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = B(i,j);
		}
	}
	data_initialized_ = true;	
}

double Matrix::operator()(const size_t i, const size_t j) const 
{
	//For now it is only a local access and it assumes that different 
	//matrices are partitioned the same way
        assert(i<n_rows_local_);
	assert(j<n_cols_);
	
	return data_[i+j*n_rows_local_];
}


Matrix::Matrix(Matrix& B):n_rows_(B.getNumRows()),n_cols_(B.getNumCols()),n_rows_local_(B.getNumRowsLocal()){

	//For now it is only a local access and it assumes that different 
	//matrices are partitioned the same way
	assert( B.getDataRawPtr()!=nullptr );

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = B(i,j);
		}
	}

	data_initialized_ = true;

} 

void Matrix::zeroInitialize(){

	assert(!data_initialized_);
	assert( data_.size()==n_rows_local_*n_cols_ );

	/*for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] = 0.0;
		}
	}*/

	data_.assign(n_rows_local_*n_cols_,0);

	data_initialized_ = true;

}

 
void Matrix::identityInitialize(){

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			if(global_row_id_[i]!=j)
				data_[i+j*n_rows_local_] = 0.0;
			else
				data_[i+j*n_rows_local_] = 1.0;
		}
	}

	data_initialized_ = true;

} 

void Matrix::randomInitialize(){

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1, +1);

	assert(!data_initialized_);

	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			data_[i+j*n_rows_local_] = dis(gen);
		}
	}

	data_initialized_ = true;

} 

size_t Matrix::getNumRows() const { return n_rows_;}
size_t Matrix::getNumRowsLocal() const { return n_rows_local_;}
size_t Matrix::getNumCols() const { return n_cols_;}

std::vector<double> Matrix::getCopyData() const
{
	std::vector<double> data_copy; 
	std::copy(data_.begin(), data_.end(), data_copy.begin());

	return data_copy;

}

const double* Matrix::getDataRawPtr() const
{
	assert( data_.data()!=nullptr );

	return data_.data();

}

void Matrix::printMatrix() const
{
	assert(data_initialized_);

        int comm_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#ifdef USE_MAGMA
	if(comm_rank==0)
		std::cout<<"MAGMA version of print"<<std::endl<<std::flush;

	std::cout<<"MPI process: "<<comm_rank<<" of "<<comm_size<<std::endl<<std::flush;
	//magma_dprint(n_rows_, n_cols_, data_.get(), n_rows_);
	magma_dprint(n_rows_local_, n_cols_, this->getDataRawPtr(), n_rows_local_);
#else		
	if(comm_rank==0)
		std::cout<<"Basic implementation of print"<<std::endl<<std::flush;

	std::cout<<"MPI process: "<<comm_rank<<" of "<<comm_size<<std::endl<<std::flush;
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_; ++i) {
			//*(data_ + i + j*n_rows_) = dis(gen);
			std::cout<< data_[i+j*n_rows_]<< "\t"<<std::flush;
		}
		std::cout<<"\n"<<std::endl<<std::flush;
	}

#endif

} 

void Matrix::computeFrobeniusNorm()
{

// Currently does not support MPI
	assert(data_.data() != nullptr);
	assert(data_initialized_);

	magma_init();
	double *dA;
	size_t ldda = magma_roundup(n_rows_, 32);
	magma_dmalloc( &dA, ldda*n_cols_ );
	assert( dA != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	//The computation of the infinity norm is not supported yet in MAGMA
	//But we know that ||A||_F <= sqrt( ||A||_1 * ||A||_inf )
	magma_norm_t one_norm = MagmaOneNorm;
	magma_norm_t inf_norm = MagmaInfNorm;
	double *dwork;
	magma_dmalloc( &dwork, ldda );
	double one_norm_value = magmablas_dlange (one_norm, n_rows_, n_cols_, dA, ldda, dwork, ldda, queue);
	double inf_norm_value = magmablas_dlange (inf_norm, n_rows_, n_cols_, dA, ldda, dwork, ldda, queue);
	std::cout<<"Computed One Norm: "<<one_norm_value<<std::endl;
	std::cout<<"Computed Inf Norm: "<<inf_norm_value<<std::endl;
	std::cout<<"Computed Upper Bound for Frobenius Norm: "<<std::sqrt(one_norm_value * inf_norm_value)<<std::endl;
	magma_finalize();

}


void Matrix::scaleMatrix(double scale_factor)
{
	std::transform(data_.begin(), data_.end(), data_.begin(),
                   [scale_factor](double alpha){ return scale_factor * alpha; });
}

void Matrix::orthogonalize(unsigned int max_iter, double tol)
{
	size_t m = n_cols_;
	size_t n = n_cols_;
	size_t k = n_rows_local_;
	double alpha = 1.0;
	double beta = 0.0;
	double* hC = new double[n_cols_ * n_cols_];
	assert( hC != nullptr );
	/*double* hA = this->getCopyRawPtrData();
	double* hB = this->getCopyRawPtrData();*/
	size_t lda = n_rows_local_;
	size_t ldb = n_rows_local_;
	size_t ldc = n_cols_;

        double* hCsum  = new double[n_cols_ * n_cols_];

#ifdef USE_MAGMA
	magma_init();

        magma_trans_t transA = MagmaTrans;
	magma_trans_t transB = MagmaNoTrans;
	double *dA, *dB, *dC;

	size_t ldda = magma_roundup(n_rows_local_, 32);
	size_t lddb = magma_roundup(n_rows_local_, 32);
	size_t lddc = magma_roundup(n_cols_, 32);

	magma_dmalloc( &dA, ldda*n_cols_ );
	magma_dmalloc( &dB, lddb*n_cols_ );
	magma_dmalloc( &dC, lddc*n_cols_ );

	assert( dA != nullptr );
	assert( dB != nullptr );
	assert( dC != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	// copy A to dA
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), lda, dA, ldda, queue );
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), ldb, dB, lddb, queue );

	/*std::cout<<"Printing hA:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	std::cout<<"Printing hB:"<<std::endl;
	magma_dprint(n_rows_, n_cols_, hB, n_rows_);
	std::cout<<"Printing dA:"<<std::endl;
	magma_dprint_gpu(ldda, n_cols_, dA, ldda, queue);
	std::cout<<"Printing dB:"<<std::endl;
	magma_dprint_gpu(lddb, n_cols_, dB, lddb, queue);*/
	magmablas_dgemm(transA,transB,m,n,k,alpha,dA,ldda,dB,lddb,beta,dC,lddc,queue);	
	//std::cout<<"Printing dC:"<<std::endl;
	//magma_dprint_gpu(lddc, n_cols_, dC, lddc, queue);
	magma_dgetmatrix( n_cols_, n_cols_, dC, lddc, hC, ldc, queue );
	/*std::cout<<"Printing hC:"<<std::endl;
	magma_dprint(n_cols_, n_cols_, hC, n_cols_);*/
        
        // sum hC over all processors
        
        MPI_Allreduce(hC, hCsum, n_cols_*n_cols_ , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	/*std::cout<<"Printing hCsum:"<<std::endl;
	magma_dprint(n_cols_, n_cols_, hCsum, n_cols_);*/
        magma_dsetmatrix( n_cols_, n_cols_, hCsum, ldc, dC, lddc, queue );
	/*std::cout<<"Printing dC after MPI_Allreduce SUM:"<<std::endl;
	magma_dprint_gpu(lddc, n_cols_, dC, lddc, queue);*/
 
	unsigned int count_iter = 0;
	double relative_residual = 1.0;

	magma_norm_t one_norm = MagmaOneNorm;
	magma_norm_t inf_norm = MagmaInfNorm;
	double *dwork;
	magma_dmalloc( &dwork, lddc );
	double one_norm_value = magmablas_dlange (one_norm, n_cols_, n_cols_, dC, lddc, dwork, lddc, queue);
	double inf_norm_value = magmablas_dlange (inf_norm, n_cols_, n_cols_, dC, lddc, dwork, lddc, queue);
	//std::cout<<"Computed Upper Bound for Frobenius Norm of A^T*A: "<<sqrt(one_norm_value * inf_norm_value)<<std::endl;

	//Implementation of Schulz iteration

	double* dI;
	double *dY, *dYaux;
	double *dZ, *dZaux;
	double* dZY;
	double* dIntermediate;
	magma_dmalloc( &dI, lddc*n_cols_ );
	magma_dmalloc( &dY, lddc*n_cols_ );
	magma_dmalloc( &dZ, lddc*n_cols_ );
	magma_dmalloc( &dYaux, lddc*n_cols_ );
	magma_dmalloc( &dZaux, lddc*n_cols_ );
	magma_dmalloc( &dZY, lddc*n_cols_ );
	magma_dmalloc( &dIntermediate, lddc*n_cols_ );

	magmablas_dlaset(MagmaFull, lddc, n_cols_, 0.0, 1.0, dI, lddc, queue);
	magma_dcopymatrix(lddc, n_cols_, dC, lddc, dY, lddc, queue);
	magmablas_dlaset(MagmaFull, lddc, n_cols_, 0.0, 1.0, dZ, lddc, queue);

	while(count_iter<max_iter & relative_residual>tol)
	{
		//std::cout<<"Iteration count"<<count_iter<<std::endl;
		// Compute ZY
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dZ,lddc,dY,lddc,beta,dZY,lddc,queue);	
		//magma_dprint_gpu(lddc, n_cols_, dZY, lddc, queue);

		//Compute 3I-ZY
		magma_dcopymatrix(lddc, n_cols_, dZY, lddc, dIntermediate, lddc, queue);
		magmablas_dgeadd2(lddc, n_cols_, 3.0, dI, lddc, -1.0, dIntermediate, lddc, queue);

		//Compute Y(3I-ZY)
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dY,lddc,dIntermediate,lddc,beta,dYaux,lddc,queue);	

		//Compute (3I-ZY)Z
		magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,lddc,n_cols_,n_cols_,alpha,dIntermediate,lddc,dZ,lddc,beta,dZaux,lddc,queue);	

		//Rescale by 1/2
		int val = 0;
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, n_cols_, dYaux, lddc, queue, &val);
                magmablas_dlascl(MagmaFull, 0, 0, 2.0, 1.0, lddc, n_cols_, dZaux, lddc, queue, &val);
		magma_dcopymatrix(lddc, n_cols_, dYaux, lddc, dY, lddc, queue);
		magma_dcopymatrix(lddc, n_cols_, dZaux, lddc, dZ, lddc, queue);

		count_iter++;
	}

	//std::cout<<"Printing dA:"<<std::endl;
	//magma_dprint_gpu(ldda, n_cols_, dA, lda, queue);
	double* dAortho;
	magma_dmalloc( &dAortho, ldda*n_cols_ );
	magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,ldda,n_cols_,n_cols_,alpha,dA,ldda,dZ,lddc,beta,dAortho,ldda,queue);	

	magma_dgetmatrix( n_rows_local_, n_cols_, dAortho, ldda, &data_[0], lda, queue );

	/*std::cout<<"Orthogonalized matrix dAortho:"<<std::endl;
	magma_dprint_gpu(ldda, n_cols_, dAortho, lda, queue);
	std::cout<<"Inverse square root:"<<std::endl;
	magma_dprint_gpu(lddc, n_cols_, dZ, lddc, queue);*/

	magma_free(dY);
	magma_free(dZ);
	magma_free(dYaux);
	magma_free(dZaux);
	magma_free(dZY);
	magma_free(dIntermediate);

	magma_free(dA);
	magma_free(dAortho);
	magma_free(dB);
	magma_free(dC);

	magma_finalize();	
#endif
	delete[] hC;	
	delete[] hCsum;	

}

void Matrix::orthogonalityCheck()
{

	size_t m = n_cols_;
	size_t n = n_cols_;
	size_t k = n_rows_local_;
	double alpha = 1.0;
	double beta = 0.0;
	double* hC = new double[n_cols_ * n_cols_];
	assert( hC != nullptr );
	/*double* hA = this->getCopyRawPtrData();
	double* hB = this->getCopyRawPtrData();*/
	size_t lda = n_rows_local_;
	size_t ldb = n_rows_local_;
	size_t ldc = n_cols_;

        double* hCsum  = new double[n_cols_ * n_cols_];

#ifdef USE_MAGMA
	magma_init();

        magma_trans_t transA = MagmaTrans;
	magma_trans_t transB = MagmaNoTrans;
	double *dA, *dB, *dC;

	size_t ldda = magma_roundup(n_rows_local_, 32);
	size_t lddb = magma_roundup(n_rows_local_, 32);
	size_t lddc = magma_roundup(n_cols_, 32);

	magma_dmalloc( &dA, ldda*n_cols_ );
	magma_dmalloc( &dB, lddb*n_cols_ );
	magma_dmalloc( &dC, lddc*n_cols_ );

	assert( dA != nullptr );
	assert( dB != nullptr );
	assert( dC != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	// copy A to dA
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), lda, dA, ldda, queue );
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), ldb, dB, lddb, queue );

	magmablas_dgemm(transA,transB,m,n,k,alpha,dA,ldda,dB,lddb,beta,dC,lddc,queue);	
	magma_dgetmatrix( n_cols_, n_cols_, dC, lddc, hC, ldc, queue );
        
        // sum hC over all processors
        
        MPI_Allreduce(hC, hCsum, n_cols_*n_cols_ , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	/*std::cout<<"Printing hCsum:"<<std::endl;
	magma_dprint(n_cols_, n_cols_, hCsum, n_cols_);*/
        magma_dsetmatrix( n_cols_, n_cols_, hCsum, ldc, dC, lddc, queue );
	std::cout<<"Printing dC after MPI_Allreduce SUM:"<<std::endl;
	magma_dprint_gpu(lddc, n_cols_, dC, lddc, queue);

#endif

}


void Matrix::matrixSum(Matrix& B)
{

	assert(n_rows_ == B.getNumRows());
	assert(n_cols_ == B.getNumCols());

	size_t lda = n_rows_local_;
	size_t ldb = n_rows_local_;

	//double* hA = this->getCopyRawPtrData();
	//std::cout<<"Printing hA: "<<std::endl;
	//magma_dprint(n_rows_, n_cols_, hA, n_rows_);
#ifdef USE_MAGMA
        magma_init();

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	size_t ldda = magma_roundup(n_rows_local_, 32);
	size_t lddb = magma_roundup(n_rows_local_, 32);

	double *dA;
	magma_dmalloc( &dA, ldda * n_cols_ );
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), n_rows_local_, dA, ldda, queue );
	/*std::cout<<"Printing dA before sum: "<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dA, ldda, queue);*/
	double *dB;
	magma_dmalloc( &dB, lddb * n_cols_ );
	magma_dsetmatrix( n_rows_local_, n_cols_, B.getDataRawPtr(), n_rows_local_, dB, lddb, queue );
	/*std::cout<<"Printing dB: "<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dB, lddb, queue);*/
        magmablas_dgeadd (ldda, n_cols_, 1.0, dB, lddb, dA, ldda, queue);
	magma_dgetmatrix( n_rows_local_, n_cols_, dA, ldda, &data_[0], n_rows_local_, queue );
	/*std::cout<<"Printing dA after sum: "<<std::endl;
	magma_dprint_gpu(n_rows_, n_cols_, dA, ldda, queue);*/
	//std::cout<<"Printing hA: "<<std::endl;
	//magma_dprint(n_rows_, n_cols_, hA, n_rows_);
	magma_free(dA);
	magma_free(dB);

	magma_finalize();
//if MAGMA not used reimplement access operator
#else
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] += B(i,j);
		}
	}
#endif
}

#endif

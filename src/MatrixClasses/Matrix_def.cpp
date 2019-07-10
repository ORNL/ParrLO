#include <random>
#include "Matrix_decl.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
 
Matrix::Matrix(size_t n, size_t m,MPI_Comm comm ):n_rows_(n),n_cols_(m),lacomm(comm){

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);
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
	std::vector<double> data_copy(data_.size(), 0.0); 
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
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);

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

double Matrix::computeFrobeniusNorm()
{
	assert(data_.data() != nullptr);

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);

	double frobSum = 0.0;
	double frobSumAll = 0.0;
	double frobNorm = 0.0;

	std::for_each( data_.begin(),data_.end(), [&frobSum](double x){frobSum += x*x;});	
        MPI_Allreduce(&frobSum, &frobSumAll, 1 , MPI_DOUBLE, MPI_SUM, lacomm);
	frobNorm = std::sqrt(frobSumAll);
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
	size_t lda = n_rows_local_;
	size_t ldb = n_rows_local_;
	size_t ldc = n_cols_;

#ifdef USE_MAGMA

        magma_trans_t transA = MagmaTrans;
	magma_trans_t transB = MagmaNoTrans;
	double *dA, *dB, *dC, *dZ;

	size_t ldda = magma_roundup(n_rows_local_, 32);
	size_t lddb = magma_roundup(n_rows_local_, 32);
	size_t lddc = magma_roundup(n_cols_, 32);
	size_t lddz = magma_roundup(n_cols_, 32);

	magma_dmalloc( &dA, ldda*n_cols_ );
	magma_dmalloc( &dB, lddb*n_cols_ );
	magma_dmalloc( &dC, lddc*n_cols_ );
	magma_dmalloc( &dZ, lddc*n_cols_ );

	assert( dA != nullptr );
	assert( dB != nullptr );
	assert( dC != nullptr );
	assert( dZ != nullptr );

	magma_queue_t queue;
	int device;
	magma_getdevice( &device );
	magma_queue_create( device, &queue );

	// copy A to dA
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), lda, dA, ldda, queue );
	magma_dsetmatrix( n_rows_local_, n_cols_, this->getDataRawPtr(), ldb, dB, lddb, queue );

	magmablas_dgemm(transA,transB,m,n,k,alpha,dA,ldda,dB,lddb,beta,dC,lddc,queue);	
	magma_dgetmatrix( n_cols_, n_cols_, dC, lddc, hC, ldc, queue );

	std::vector<double> hCvector(hC, hC + n_cols_*n_cols_); 
	Replicated AtA(hCvector, lacomm);
	AtA.Schulz(10, 0.01);
	std::vector<double> Z = AtA.getCopyData();
	magma_dsetmatrix( n_cols_, n_cols_, &Z[0], n_cols_, dZ, lddz, queue );
        
	double* dAortho;
	magma_dmalloc( &dAortho, ldda*n_cols_ );
	magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,ldda,n_cols_,n_cols_,alpha,dA,ldda,dZ,lddc,beta,dAortho,ldda,queue);	

	magma_dgetmatrix( n_rows_local_, n_cols_, dAortho, ldda, &data_[0], lda, queue );

	magma_free(dZ);

	magma_free(dA);
	magma_free(dAortho);
	magma_free(dB);
	magma_free(dC);

#endif
	delete[] hC;	

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
//	magma_init();

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
        MPI_Allreduce(hC, hCsum, n_cols_*n_cols_ , MPI_DOUBLE, MPI_SUM, lacomm);
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
//        magma_init();

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

//	magma_finalize();
//if MAGMA not used reimplement access operator
#else
	for (size_t j = 0; j < n_cols_; ++j) {
        	for (size_t i = 0; i < n_rows_local_; ++i) {
			data_[i+j*n_rows_local_] += B(i,j);
		}
	}
#endif
}


#include "MatrixClasses/Matrix_decl.hpp"
#include <mpi.h>
#include <iostream>
#include <unistd.h>

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main(int argc, char **argv)
{

        int i = MPI_Init(&argc,&argv);
 
        if( i != MPI_SUCCESS){
        }else{

        MPI_Comm lacomm;      
       
        MPI_Comm_dup(MPI_COMM_WORLD, &lacomm); 

	MPI_Barrier(lacomm);
        std::cout<<"MPI SUCCESS"<<i<<std::endl<<std::flush;

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);
  
#ifdef USE_MAGMA
        magma_init();
#else

#endif
	int matrix_dim = 2;
       
        Matrix A(matrix_dim,matrix_dim,lacomm);
        Matrix B(matrix_dim,matrix_dim,lacomm);
	MPI_Barrier(lacomm);

//	A.zeroInitialize();
	A.randomInitialize();
        A.printMatrix();

	sleep(3);
	MPI_Barrier(lacomm);

	if(comm_rank==0)
		std::cout<<"Rescaling of A"<<std::endl<<std::flush;

	A.scaleMatrix(0.01);
        A.printMatrix();

	sleep(3);
	MPI_Barrier(lacomm);

	if(comm_rank==0)
		std::cout<<"Initialization of B"<<std::endl<<std::flush;

	B.identityInitialize();
        B.printMatrix();
	sleep(3);
	
	MPI_Barrier(lacomm);

	if(comm_rank==0)
		std::cout<<"A+B"<<std::endl<<std::flush;

        A.matrixSum(B);
	MPI_Barrier(lacomm);
	sleep(3);
        A.printMatrix();
	sleep(3);
//	A.computeFrobeniusNorm();
	//A.matrixSum();
	//

	A.sumAllProcesses();
	A.orthogonalize(10, 0.1);

	MPI_Barrier(lacomm);
	sleep(3);
	if(comm_rank==0)
		std::cout<<"Orthogonalized A"<<std::endl<<std::flush;
        A.printMatrix();

	if(comm_rank==0)
		std::cout<<"Orthogonality check A"<<std::endl<<std::flush;
	A.orthogonalityCheck(); 

	double frobeniusNorm = A.computeFrobeniusNorm();
	if(comm_rank==0)
		std::cout<<"Frobenius norm of orthogonalized matrix: "<<frobeniusNorm<<std::endl<<std::flush;
  
       MPI_Comm_free(&lacomm);
       }
#ifdef USE_MAGMA
       magma_finalize();
#else

#endif
       MPI_Finalize();

	return 0;
}

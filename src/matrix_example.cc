#include "MatrixClasses/Matrix_def.hpp"
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

	MPI_Barrier(MPI_COMM_WORLD);
        std::cout<<"MPI SUCCESS"<<i<<std::endl<<std::flush;

        int comm_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int matrix_dim = 10;
       
        Matrix A(matrix_dim,matrix_dim);
        Matrix B(matrix_dim,matrix_dim);

	if(comm_rank==0)
		std::cout<<"Initialization of A"<<std::endl<<std::flush;

	sleep(3);
	MPI_Barrier(MPI_COMM_WORLD);

//	A.zeroInitialize();
	A.randomInitialize();
        //A.printMatrix();

	sleep(3);
	MPI_Barrier(MPI_COMM_WORLD);

	if(comm_rank==0)
		std::cout<<"Rescaling of A"<<std::endl<<std::flush;

	A.scaleMatrix(0.01);
        A.printMatrix();

	sleep(3);
	MPI_Barrier(MPI_COMM_WORLD);

	if(comm_rank==0)
		std::cout<<"Initialization of B"<<std::endl<<std::flush;

	B.identityInitialize();
        //B.printMatrix();
	sleep(3);
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(comm_rank==0)
		std::cout<<"A+B"<<std::endl<<std::flush;

        A.matrixSum(B);
	MPI_Barrier(MPI_COMM_WORLD);
	sleep(3);
        A.printMatrix();
	sleep(3);
//	A.computeFrobeniusNorm();
	//A.matrixSum();
	//

	A.orthogonalize(5, 0.1);

	MPI_Barrier(MPI_COMM_WORLD);
	sleep(3);
	if(comm_rank==0)
		std::cout<<"Orthogonalized A"<<std::endl<<std::flush;
        //A.printMatrix();

	if(comm_rank==0)
		std::cout<<"Orthogonality check A"<<std::endl<<std::flush;
	A.orthogonalityCheck();

       }
       MPI_Finalize();

	return 0;
}

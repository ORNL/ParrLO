#include "MatrixClasses/Matrix_def.hpp"
#include <mpi.h>
#include <iostream>

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main(int argc, char **argv)
{

        int i = MPI_Init(&argc,&argv);
 
        if( i != MPI_SUCCESS)
        }else{

        std::cout<<"MPI SUCCESS"<<i<<std::endl;

       
        Matrix A(5,5);
        Matrix B(5,5);

//	A.zeroInitialize();
	A.randomInitialize();
        //A.printMatrix();
	A.scaleMatrix(0.01);
        //A.printMatrix();

	B.identityInitialize();
        //B.printMatrix();
	
        A.matrix_sum(B);
        //A.printMatrix();
//	A.computeFrobeniusNorm();
	//A.matrix_sum();
	A.orthogonalize(10, 0.1);
        //A.printMatrix();

       }
       MPI_Finalize();

	return 0;
}

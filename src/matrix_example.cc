#include "MatrixClasses/Matrix_def.hpp"

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main()
{
        Matrix A(5,5);
        Matrix B(5,5);

	//A.ZeroInitialize();
	A.randomInitialize();
        //A.printMatrix();
	A.scaleMatrix(0.01);
        //A.printMatrix();

	B.identityInitialize();
        //B.printMatrix();
	
        A.matrix_sum(B);
        //A.printMatrix();
	A.computeFrobeniusNorm();
	//A.matrix_sum();
	A.orthogonalize(10, 0.1);
        //A.printMatrix();

	return 0;
}

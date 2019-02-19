#include "MatrixClasses/Matrix_def.hpp"

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main()
{
        Matrix A(5,2);

	//A.ZeroInitialize();
	A.RandomInitialize();
        //A.printMatrix();
        double* B = A.getCopyData();
	A.matrix_sum();
	A.orthogonalize();

	return 0;
}

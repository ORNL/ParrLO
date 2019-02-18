#include "MatrixClasses/Matrix_def.hpp"

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

int main()
{
        Matrix A(5,5);

	A.ZeroInitialize();
        A.printMatrix();
        double* B = A.getCopyData();

	return 0;
}

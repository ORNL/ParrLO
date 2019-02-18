#include "MatrixClasses/Matrix_def.hpp"

int main()
{
        Matrix A(5,5);

	A.ZeroInitialize();
        A.printMatrix();
        double* B = A.getCopyData();

	return 0;
}

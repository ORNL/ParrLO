#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        return 1;
    }

    int mype = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    std::cout << "Hello from task " << mype << std::endl;

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI_Finalize() failed!!!" << std::endl;
        return 1;
    }

    // return 0 for SUCCESS
    return 0;
}

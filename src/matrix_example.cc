#include "MatrixClasses/Matrix_decl.hpp"
#include "MatrixClasses/Timer.hpp"
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unistd.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifndef USE_MAGMA
#define USE_MAGMA
#endif

// Usage:
// mpirun -np $ncpus matrix_example -c input.cfg
//
// where ncpus is the number of MPI tasks, and the content of the input file
// input.cfg looks like:
//
// [Matrix]
// nrows=32
// ncols=4

int main(int argc, char** argv)
{

    int i = MPI_Init(&argc, &argv);

    if (i != MPI_SUCCESS)
    {
    }
    else
    {

        MPI_Comm lacomm;

        MPI_Comm_dup(MPI_COMM_WORLD, &lacomm);

        std::cout << "MPI SUCCESS" << i << std::endl;

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);

        std::vector<int> idata;

        // read run time-parameters from PE0
        if (comm_rank == 0)
        {
            po::variables_map vm;
            std::string config_file;

            po::options_description generic("Generic options");
            generic.add_options()(
                "config,c", po::value<std::string>(&config_file)
                                ->default_value("input.cfg"));

            // run-time options to be read from config file
            po::options_description config("Configuration");
            config.add_options()("Matrix.nrows", po::value<int>()->required(),
                "number of matrix rows")("Matrix.ncols",
                po::value<int>()->required(), "number of matrix columns");

            po::options_description cmdline_options;
            cmdline_options.add(generic);

            po::options_description config_file_options;
            config_file_options.add(config);

            po::options_description visible("Allowed options");
            visible.add(generic).add(config);

            store(po::command_line_parser(argc, argv)
                      .options(cmdline_options)
                      .run(),
                vm);
            notify(vm);

            std::ifstream ifs(config_file.c_str());
            if (!ifs)
            {
                std::cerr << "Can not open config file: " << config_file
                          << "\n";
                return -1;
            }
            else
            {
                store(parse_config_file(ifs, config_file_options), vm);
                notify(vm);
            }

            // save run-time options in a vector that can be used
            // to bcast to other MPI tasks
            idata.push_back(vm["Matrix.nrows"].as<int>());
            idata.push_back(vm["Matrix.ncols"].as<int>());
        }

        // broadcast input parameters to all MPI tasks
        int nidata = idata.size();
        MPI_Bcast(&nidata, 1, MPI_INT, 0, lacomm);

        if (comm_rank != 0) idata.resize(nidata);
        MPI_Bcast(idata.data(), nidata, MPI_INT, 0, lacomm);

        std::string name = "matrix_example";
        Timer matrix_time(name);
        matrix_time.start();

#ifdef USE_MAGMA
        magma_init();
#else

#endif
        // matrix_time.start();

        int nrows = idata[0];
        int ncols = idata[1];
        if (comm_rank == 0)
        {
            std::cout << "Matrix size: " << nrows << "x" << ncols << std::endl;
        }

        Matrix A(nrows, ncols, lacomm);
        Matrix B(nrows, ncols, lacomm);

        //	A.zeroInitialize();
        A.randomInitialize();
        A.printMatrix();

        if (comm_rank == 0) std::cout << "Rescaling of A" << std::endl;

        A.scaleMatrix(0.01);
        A.printMatrix();

        if (comm_rank == 0) std::cout << "Initialization of B" << std::endl;

        B.identityInitialize();
        B.printMatrix();

        if (comm_rank == 0) std::cout << "A+B" << std::endl;

        A.matrixSum(B);
        A.printMatrix();

        A.orthogonalize(10, 0.1);

        if (comm_rank == 0) std::cout << "Orthogonalized A" << std::endl;
        A.printMatrix();

        if (comm_rank == 0) std::cout << "Orthogonality check A" << std::endl;

        double departure_from_orthogonality = 0.0;
        departure_from_orthogonality        = A.orthogonalityCheck();

        if (comm_rank == 0)
            std::cout << "Departure from orthogonality: "
                      << departure_from_orthogonality << std::endl;

        MPI_Comm_free(&lacomm);

        matrix_time.stop();

        matrix_time.print(std::cout);
    }
#ifdef USE_MAGMA
    magma_finalize();
#else

#endif

    MPI_Finalize();

    return 0;
}

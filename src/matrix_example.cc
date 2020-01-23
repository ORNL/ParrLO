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
// [Rescaling]
// rescaling=0.01
// [Schulz_iteration]
// max_iterations=100
// tolerance=1e-4

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

        // std::cout << "MPI SUCCESS" << i << std::endl;

        int comm_rank, comm_size;
        MPI_Comm_rank(lacomm, &comm_rank);
        MPI_Comm_size(lacomm, &comm_size);

        std::vector<int> idata;
        double irescaling;
        int imax_iterations;
        double itolerance;

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
                po::value<int>()->required(), "number of matrix columns")(
                "Rescaling.rescaling", po::value<double>()->required(),
                "rescaling for perturbation from orthogonality")(
                "Schulz_iteration.max_iterations", po::value<int>()->required(),
                "maximum number of iterations allowed for Schulz algorithm")(
                "Schulz_iteration.tolerance", po::value<double>()->required(),
                "stopping tolerance for Schulz algorithm");

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
            irescaling      = vm["Rescaling.rescaling"].as<double>();
            imax_iterations = vm["Schulz_iteration.max_iterations"].as<int>();
            itolerance      = vm["Schulz_iteration.tolerance"].as<double>();
        }

        // broadcast input parameters to all MPI tasks
        int nidata = idata.size();
        MPI_Bcast(&nidata, 1, MPI_INT, 0, lacomm);

        if (comm_rank != 0) idata.resize(nidata);
        MPI_Bcast(idata.data(), nidata, MPI_INT, 0, lacomm);
        MPI_Bcast(&irescaling, 1, MPI_DOUBLE, 0, lacomm);
        MPI_Bcast(&imax_iterations, 1, MPI_INT, 0, lacomm);
        MPI_Bcast(&itolerance, 1, MPI_DOUBLE, 0, lacomm);

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

        A.gaussianColumnsInitialize(0.8);
        // A.activateRescaling();

        // Perform the check on the departure from orthogonality before
        // re-orthogonalizing
        double departure_from_orthogonality = 0.0;
        departure_from_orthogonality        = A.orthogonalityCheck();

        if (comm_rank == 0)
            std::cout
                << "Departure from orthogonality before re-orthogonalizing: "
                << departure_from_orthogonality << std::endl;

        std::string name_ortho = "orthogonalization";
        Timer orthogonalization_timer(name_ortho);

        orthogonalization_timer.start();
        A.orthogonalize_iterative_method(imax_iterations, itolerance);
        orthogonalization_timer.stop();

        if (comm_rank == 0) std::cout << "Orthogonalized A" << std::endl;
        // A.printMatrix();

        if (comm_rank == 0) std::cout << "Orthogonality check A" << std::endl;

        // Perform the check on the departure from orthogonality after
        // re-orthogonalizing
        departure_from_orthogonality = A.orthogonalityCheck();

        if (comm_rank == 0)
            std::cout
                << "Departure from orthogonality after re-orthogonalizing: "
                << departure_from_orthogonality << std::endl;

        MPI_Comm_free(&lacomm);

        matrix_time.stop();

        matrix_time.print(std::cout);
        orthogonalization_timer.print(std::cout);

        Replicated::printTimers(std::cout);
        Matrix::printTimers(std::cout);
    }
#ifdef USE_MAGMA
    magma_finalize();
#else

#endif

    MPI_Finalize();

    return 0;
}

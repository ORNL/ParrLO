#include "MatrixClasses/Matrix_decl.hpp"
#include "MatrixClasses/Timer.hpp"
#include "warmup.h"

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <unistd.h>

#ifdef NCCL_COMM
#include "nccl.h"
#endif

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
// [ColumnsType]
// wavefunctions_type = gaussian
//                    = hat
// [ColumnsCenter]
// displacement = 0.5
// [ColumnsWidth]
// wavefunctions_width = 0.1
// [Rescaling]
// rescaling=0.01
// [DiagonalRescaling]
// rescaling=true
//          =false
// [Orthogonalization]
// method_type=direct_method
//            =iterative_method_single
//            =iterative_method_couple
// num_runs=50
// [Schulz_iteration]
// max_iterations=100
// tolerance=1e-4
// implementation=original
//               =delta
// ntasks = 2
// [Convergence]
// convergence_check="relative"
//                  ="absolute"
// frequency_convergence_check = 1

int main(int argc, char** argv)
{

    int i = MPI_Init(&argc, &argv);

    if (i != MPI_SUCCESS)
    {
        std::cerr << "MPI_Init failed!" << std::endl;
        return 1;
    }
    else
    {
        int comm_rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        std::vector<int> idata;
        std::string iwavefunctions_type;
        double iwavefunctions_center_displacement = 0.0;
        double iwavefunctions_width               = 0.8;
        double irescaling;
        int inum_runs;
        bool idiagonal_rescaling;
        std::string iortho_type;
        int imax_iterations;
        int intasks;
        double itolerance;
        std::string iconvergence_check   = "relative";
        int ifrequency_convergence_check = 1;
        std::string iimplementation;

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
                po::value<int>()->required(),
                "number of matrix columns")("ColumnsType.wavefunctions_type",
                po::value<std::string>()->required(), "wavefunction type")(
                "ColumnsCenter.displacement", po::value<double>()->required(),
                "displacement ratio for the center of the wave functions")(
                "ColumnsWidth.wavefunctions_width",
                po::value<double>()->required(),
                "width ratio for the extension of the wave functions")(
                "Rescaling.rescaling", po::value<double>()->required(),
                "rescaling for perturbation from orthogonality")(
                "DiagonalRescaling.rescaling", po::value<bool>()->required(),
                "rescaling for Schulz iteration")(
                "Orthogonalization.method_type",
                po::value<std::string>()->required(),
                "method of orthogonalization")("Orthogonalization.num_runs",
                po::value<int>()->default_value(1), "number of runs")(
                "Schulz_iteration.max_iterations", po::value<int>()->required(),
                "maximum number of iterations allowed for Schulz algorithm")(
                "Schulz_iteration.tolerance", po::value<double>()->required(),
                "stopping tolerance for Schulz algorithm")(
                "Schulz_iteration.implementation",
                po::value<std::string>()->default_value("original"),
                "Choice of implementation")("Schulz_iteration.ntasks",
                po::value<int>()->default_value(1),
                "Number of tasks to distribute Schulz computation")(
                "Convergence.convergence_check",
                po::value<std::string>()->default_value("relative"),
                "Type of convergence check")(
                "Convergence.frequency_convergence_check",
                po::value<int>()->default_value(1),
                "Frequency to compute the convergence check");

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
            iwavefunctions_type
                = vm["ColumnsType.wavefunctions_type"].as<std::string>();
            iwavefunctions_width
                = vm["ColumnsWidth.wavefunctions_width"].as<double>();
            iwavefunctions_center_displacement
                = vm["ColumnsCenter.displacement"].as<double>();
            irescaling          = vm["Rescaling.rescaling"].as<double>();
            idiagonal_rescaling = vm["DiagonalRescaling.rescaling"].as<bool>();
            iortho_type = vm["Orthogonalization.method_type"].as<std::string>();
            inum_runs   = vm["Orthogonalization.num_runs"].as<int>();
            imax_iterations = vm["Schulz_iteration.max_iterations"].as<int>();
            itolerance      = vm["Schulz_iteration.tolerance"].as<double>();
            iimplementation
                = vm["Schulz_iteration.implementation"].as<std::string>();
            intasks = vm["Schulz_iteration.ntasks"].as<int>();
            iconvergence_check
                = vm["Convergence.convergence_check"].as<std::string>();
            ifrequency_convergence_check
                = vm["Convergence.frequency_convergence_check"].as<int>();
        }

        // broadcast input parameters to all MPI tasks
        int nidata = idata.size();
        int ret    = MPI_Bcast(&nidata, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) std::cerr << "MPI_Bcast error!" << std::endl;

        if (comm_rank != 0) idata.resize(nidata);
        MPI_Bcast(idata.data(), nidata, MPI_INT, 0, MPI_COMM_WORLD);
        int wave_string_length = iwavefunctions_type.length();
        MPI_Bcast(&wave_string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&iwavefunctions_width, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&iwavefunctions_center_displacement, 1, MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
        MPI_Bcast(&inum_runs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&irescaling, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&idiagonal_rescaling, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        int ortho_string_length = iortho_type.length();
        MPI_Bcast(&ortho_string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int convergence_string_length = iconvergence_check.length();
        MPI_Bcast(&convergence_string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // bcast "iortho_type"
        {
            std::vector<char> iortho_type_v(ortho_string_length + 1);
            if (comm_rank == 0)
                strcpy(iortho_type_v.data(), iortho_type.c_str());
            ret = MPI_Bcast(iortho_type_v.data(), iortho_type_v.size(),
                MPI_CHAR, 0, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS)
                std::cerr << "iortho_type_v: MPI_Bcast error!" << std::endl;
            else
            {
                std::string new_iortho_type(iortho_type_v.data());
                iortho_type = new_iortho_type;
            }
        }

        // bcast "iwavefunctions_type"
        {
            std::vector<char> iwavefunctions_type_v(wave_string_length + 1);
            if (comm_rank == 0)
                strcpy(
                    iwavefunctions_type_v.data(), iwavefunctions_type.c_str());
            ret = MPI_Bcast(iwavefunctions_type_v.data(),
                iwavefunctions_type_v.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS)
                std::cerr << "iortho_type_v: MPI_Bcast error!" << std::endl;
            else
            {
                std::string new_iwavefunctions_type(
                    iwavefunctions_type_v.data());
                iwavefunctions_type = new_iwavefunctions_type;
            }
        }

        // bcast "iconvergence_check"
        {
            std::vector<char> iconvergence_check_v(
                convergence_string_length + 1);
            if (comm_rank == 0)
                strcpy(iconvergence_check_v.data(), iconvergence_check.c_str());
            ret = MPI_Bcast(iconvergence_check_v.data(),
                iconvergence_check_v.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS)
                std::cerr << "iconvergence_type_v: MPI_Bcast error!"
                          << std::endl;
            else
            {
                std::string new_iconvergence_check(iconvergence_check_v.data());
                iconvergence_check = new_iconvergence_check;
            }
        }
        // bcast "iimplementation"
        {
            std::vector<char> v(convergence_string_length + 1);
            if (comm_rank == 0) strcpy(v.data(), iimplementation.c_str());
            ret = MPI_Bcast(v.data(), v.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS)
                std::cerr << "iimplementation`: MPI_Bcast error!" << std::endl;
            else
            {
                std::string new_iimplementation(v.data());
                iimplementation = new_iimplementation;
            }
        }

        MPI_Bcast(&imax_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&intasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&itolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ifrequency_convergence_check, 1, MPI_INT, 0, MPI_COMM_WORLD);

        Timer matrix_time("matrix_ortho");
        matrix_time.start();

#ifdef NCCL_COMM
        ncclUniqueId id;
        ncclComm_t nccl_world_comm;

        if (comm_rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // create NCCL communicator
        ncclCommInitRank(&nccl_world_comm, comm_size, id, comm_rank);
#else
        int nccl_world_comm = 0;
#endif

#ifdef USE_MAGMA
        magma_init();

        if (comm_rank == 0) magma_print_environment();
#endif

        warmup_MPI_pt2pt(MPI_COMM_WORLD);

#ifdef NCCL_COMM
        warmup_NCCL(nccl_world_comm);
#endif

        int nrows = idata[0];
        int ncols = idata[1];
        if (comm_rank == 0)
        {
            std::cout << "Matrix size: " << nrows << "x" << ncols << std::endl;
        }

        Matrix A(nrows, ncols, MPI_COMM_WORLD, nccl_world_comm);

        double standard_deviation = iwavefunctions_width;
        double support_length     = iwavefunctions_width;

        if (iwavefunctions_type == "gaussian")
            A.gaussianColumnsInitialize(
                standard_deviation, iwavefunctions_center_displacement);
        else if (iwavefunctions_type == "hat")
            A.hatColumnsInitialize(
                support_length, iwavefunctions_center_displacement);

        for (int step = 0; step < inum_runs; step++)
        {
            if (step) A.transferDataCPUtoGPU();
            // Perform the check on the departure from orthogonality before
            // re-orthogonalizing
            double departure_from_orthogonality = 0.0;
            departure_from_orthogonality        = A.orthogonalityCheck();

            if (comm_rank == 0)
                std::cout << "Step = " << step << "/" << inum_runs << "\n"
                          << "Departure from orthogonality before "
                             "re-orthogonalizing: "
                          << departure_from_orthogonality << std::endl;

            int count_iter = A.orthogonalize(iortho_type, idiagonal_rescaling,
                imax_iterations, itolerance, iimplementation, intasks,
                iconvergence_check, ifrequency_convergence_check);

            if (comm_rank == 0) std::cout << "Orthogonalized A" << std::endl;
            if (comm_rank == 0 && count_iter > 0)
                std::cout << "Iterative solve took " << count_iter
                          << " iterations" << std::endl;

            // A.printMatrix();

            if (comm_rank == 0)
                std::cout << "Orthogonality check A" << std::endl;

            // Perform the check on the departure from orthogonality after
            // re-orthogonalizing
            departure_from_orthogonality = A.orthogonalityCheck();

            if (comm_rank == 0)
                std::cout
                    << "Departure from orthogonality after re-orthogonalizing: "
                    << departure_from_orthogonality << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        matrix_time.stop();

        matrix_time.print(std::cout);

        Replicated::printTimers(std::cout);
        Matrix::printTimers(std::cout);
    }

#ifdef USE_MAGMA
    magma_finalize();
#endif

#ifdef NCCL_COMM
//    ncclCommDestroy(nccl_world_comm);
#endif

    MPI_Finalize();

    return 0;
}

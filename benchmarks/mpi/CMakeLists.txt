set(SOURCES_MPI_HOST ${CMAKE_SOURCE_DIR}/benchmarks/mpi/test_mpi_host.cc ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_MPI_DEVICE ${CMAKE_SOURCE_DIR}/benchmarks/mpi/test_mpi_device.cc ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
if(${NCCL_COMM})
  set(SOURCES_MPI_NCCL ${CMAKE_SOURCE_DIR}/benchmarks/mpi/test_mpi_nccl.cc ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
endif ()

set(MAIN_MPI_HOST test_mpi_host)
add_executable(${MAIN_MPI_HOST} ${SOURCES_MPI_HOST})
set(MAIN_MPI_DEVICE test_mpi_device)
add_executable(${MAIN_MPI_DEVICE} ${SOURCES_MPI_DEVICE})
if(${NCCL_COMM})
  set(MAIN_MPI_NCCL test_mpi_nccl)
  add_executable(${MAIN_MPI_NCCL} ${SOURCES_MPI_NCCL})
endif ()

foreach(X ${MAIN_MPI_HOST} ${MAIN_MPI_DEVICE} ${MAIN_MPI_NCCL})
  target_link_libraries(${X} PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(${X} PRIVATE ${MPI_CXX_LIBRARIES})
  target_link_libraries(${X} PRIVATE Boost::program_options Boost::boost)
  target_include_directories(${X} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endforeach()

if (${NCCL_COMM})
  foreach(X ${MAIN_MPI_NCCL})
    target_link_libraries(${X} PRIVATE ${NCCL_LIBRARY})
  endforeach()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  foreach(X ${MAIN_MPI_HOST} ${MAIN_MPI_DEVICE} ${MAIN_MPI_NCCL})
    target_link_libraries(${X} PUBLIC OpenMP::OpenMP_CXX)
  endforeach()
endif()

if (${BUILD_FOR_GPU})
  foreach(X ${MAIN_MPI_HOST} ${MAIN_MPI_DEVICE} ${MAIN_MPI_NCCL})
    target_link_libraries(${X} PRIVATE PkgConfig::MAGMA)
    set_target_properties(${X} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD_REQUIRED ON
      CUDA_STANDARD 11
      CUDA_STANDARD_REQUIRED ON
      CXX_STANDARD_REQUIRED ON
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON)
  endforeach()
endif()

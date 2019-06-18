cmake_minimum_required(VERSION 3.1)
project (ParallelLinearAlgebra)


set (CMAKE_CXX_STANDARD 11)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(${CMAKE_SOURCE_DIR}/cmake/checkCFortranFunctionExists.cmake)

find_package(BLAS)
if(BLAS_FOUND)
  message(STATUS
    "BLAS was found:\n"
    "  BLAS_LIBRARIES: ${BLAS_LIBRARIES}\n"
  )
endif()

if(NOT (BLAS_FOUND OR NOBLAS))
  message(FATAL_ERROR "Could not find BLAS library.")
endif()

set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
list(APPEND LINK_LIBRARIES ${BLAS_LIBRARIES})

if(BLAS_FOUND)
  check_C_Fortran_function_exists(dscal C_DSCAL REQUIRED)
  add_definitions(
    -DC_DSCAL=${C_DSCAL})
endif()

set(BUILD_FOR_GPU FALSE CACHE BOOL "Whether to build for GPU")
if(BUILD_FOR_GPU)
  FIND_PACKAGE(CUDA)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Could not find CUDA libraries")
  endif()
  include_directories(${CUDA_INCLUDE_DIRS})
  FIND_PACKAGE(MAGMA)
  if (NOT MAGMA_FOUND)
    message(FATAL_ERROR "Could not find MAGMA library")
  endif()
endif()

if (MAGMA_FOUND)
  add_definitions(-DUSE_MAGMA)
  message(STATUS
    "MAGMA was found:\n"
    "  MAGMA_INCLUDE_DIRS: ${MAGMA_INCLUDE_DIRS}\n"
    "  MAGMA_LIBRARY_DIRS: ${MAGMA_LIBRARY_DIRS}\n"
    "  MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}"
  )
  include_directories(${MAGMA_INCLUDE_DIRS})
  list(APPEND LINK_LIBRARIES ${MAGMA_LIBRARIES})
endif()

find_package(MPI REQUIRED)
if (${MPI_CXX_FOUND})
  include_directories(${MPI_INCLUDE_PATH})
else(${MPI_CXX_FOUND})
  message(FATAL_ERROR "Required MPI package not found.")
endif (${MPI_CXX_FOUND})
list(APPEND LINK_LIBRARIES ${MPI_CXX_LIBRARIES})

enable_testing()

add_subdirectory(src)

add_subdirectory(tests)

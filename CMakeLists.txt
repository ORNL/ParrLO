cmake_minimum_required(VERSION 3.1)
project (ParallelLinearAlgebra)

set (CMAKE_CXX_STANDARD 11)

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

if(BLAS_FOUND)
  check_C_Fortran_function_exists(dscal C_DSCAL REQUIRED)
  add_definitions(
    -DC_DSCAL=${C_DSCAL})
endif()

add_subdirectory(src)


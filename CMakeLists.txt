cmake_minimum_required(VERSION 3.11)
project (ParallelLinearAlgebra LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(${CMAKE_SOURCE_DIR}/cmake/checkCFortranFunctionExists.cmake)

find_package(BLAS REQUIRED)
if(${BLAS_FOUND})
  message(STATUS
    "BLAS was found:\n"
    "  BLAS_LIBRARIES: ${BLAS_LIBRARIES}\n"
  )
endif()

if(${BLAS_FOUND})
  # Add BLAS LIBRARIES to the path searched by check_C_Fortran_function_exists
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_C_Fortran_function_exists(dscal C_DSCAL REQUIRED)
  add_definitions(-DC_DSCAL=${C_DSCAL})
endif()

set(BUILD_FOR_GPU FALSE CACHE BOOL "Whether to build for GPU")
if(${BUILD_FOR_GPU})
  enable_language(CUDA)

  find_package(MAGMA REQUIRED)
endif()

if(${MAGMA_FOUND})
  add_definitions(-DUSE_MAGMA)
  message(STATUS
    "MAGMA was found:\n"
    "  MAGMA_INCLUDE_DIRS: ${MAGMA_INCLUDE_DIRS}\n"
    "  MAGMA_LIBRARY_DIRS: ${MAGMA_LIBRARY_DIRS}\n"
    "  MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}"
  )
endif()

find_package(MPI REQUIRED)

set(Boost_COMPONENTS
  program_options
  )
find_package(Boost 1.65 REQUIRED COMPONENTS ${Boost_COMPONENTS})

# clang-format (optional)
set(PLA_WITH_CLANG_FORMAT FALSE CACHE BOOL "Indent code with clang-format")
if(${PLA_WITH_CLANG_FORMAT})
  find_package(CLANG_FORMAT)
  if(${CLANG_FORMAT_FOUND})
    message(STATUS "Indent with clang-format")
    file(GLOB_RECURSE FORMAT_SOURCES src/*.cc src/*.h src/*.cpp src/*.hpp
      tests/*.cc tests/*.h tests/*.cpp tests/*.hpp)
    add_custom_target(format
      COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
      DEPENDS ${FORMAT_SOURCES})
  endif()
endif()

enable_testing()

add_subdirectory(src)

add_subdirectory(tests)

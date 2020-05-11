find_package(PythonInterp 3.2 REQUIRED)

set(SOURCES_TESTMPI ${CMAKE_SOURCE_DIR}/tests/testMPI.cc)
set(SOURCES_TESTMAGMA ${CMAKE_SOURCE_DIR}/tests/testMAGMA.cc)
set(SOURCES_TESTMAXNORM ${CMAKE_SOURCE_DIR}/tests/testMaxNormReplicated.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTSCHULZ ${CMAKE_SOURCE_DIR}/tests/testSchulz.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTSCHULZSINGLE ${CMAKE_SOURCE_DIR}/tests/testSchulzSingle.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTORTHO ${CMAKE_SOURCE_DIR}/tests/testOrtho.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Matrix_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTORTHO_RESCALE ${CMAKE_SOURCE_DIR}/tests/testOrthoRescale.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Matrix_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTSCHULZ_RESCALE ${CMAKE_SOURCE_DIR}/tests/testSchulzRescale.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTSCHULZSINGLE_RESCALE ${CMAKE_SOURCE_DIR}/tests/testSchulzSingleRescale.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTALLREDUCE ${CMAKE_SOURCE_DIR}/tests/testAllreduce.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
set(SOURCES_TESTMATMUL ${CMAKE_SOURCE_DIR}/tests/testMatMul.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)

if (${NCCL_COMM})
  set(SOURCES_TESTALLREDUCE_NCCL ${CMAKE_SOURCE_DIR}/tests/testAllreduce_nccl.cc
                        ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)
endif ()

set(MAIN_TESTMPI testMPI)
add_executable(${MAIN_TESTMPI} ${SOURCES_TESTMPI})
set(MAIN_TESTMAGMA testMAGMA)
add_executable(${MAIN_TESTMAGMA} ${SOURCES_TESTMAGMA})
set(MAIN_TESTMAXNORM testMaxNormReplicated)
add_executable(${MAIN_TESTMAXNORM} ${SOURCES_TESTMAXNORM})
set(MAIN_TESTSCHULZ testSchulz)
add_executable(${MAIN_TESTSCHULZ} ${SOURCES_TESTSCHULZ})
set(MAIN_TESTSCHULZSINGLE testSchulzSingle)
add_executable(${MAIN_TESTSCHULZSINGLE} ${SOURCES_TESTSCHULZSINGLE})
set(MAIN_TESTORTHO testOrtho)
add_executable(${MAIN_TESTORTHO} ${SOURCES_TESTORTHO})
set(MAIN_TESTORTHO_RESCALE testOrthoRescale)
add_executable(${MAIN_TESTORTHO_RESCALE} ${SOURCES_TESTORTHO_RESCALE})
set(MAIN_TESTALLREDUCE testAllreduce)
add_executable(${MAIN_TESTALLREDUCE} ${SOURCES_TESTALLREDUCE})
set(MAIN_TESTMATMUL testMatMul)
add_executable(${MAIN_TESTMATMUL} ${SOURCES_TESTMATMUL})

if (${NCCL_COMM})
  set(MAIN_TESTALLREDUCE_NCCL testAllreduce_nccl)
  add_executable(${MAIN_TESTALLREDUCE_NCCL} ${SOURCES_TESTALLREDUCE_NCCL})
endif ()

foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA} ${MAIN_TESTMAXNORM} ${MAIN_TESTSCHULZ} ${MAIN_TESTSCHULZSINGLE} ${MAIN_TESTORTHO} ${MAIN_TESTORTHO_RESCALE} ${MAIN_TESTALLREDUCE} ${MAIN_TESTALLREDUCE_NCCL} ${MAIN_TESTMATMUL}) 
  target_link_libraries(${X} PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(${X} PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(${X} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_include_directories(${X} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()

if (${BUILD_FOR_GPU})
  foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA} ${MAIN_TESTMAXNORM} ${MAIN_TESTSCHULZ} ${MAIN_TESTSCHULZSINGLE} ${MAIN_TESTORTHO} ${MAIN_TESTORTHO_RESCALE} ${MAIN_TESTALLREDUCE} ${MAIN_TESTALLREDUCE_NCCL}  ${MAIN_TESTMATMUL})
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
else()
  foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA} ${MAIN_TESTMAXNORM} ${MAIN_TESTSCHULZ} ${MAIN_TESTSCHULZSINGLE} ${MAIN_TESTORTHO} ${MAIN_TESTORTHO_RESCALE})
    set_target_properties(${X} PROPERTIES
      CXX_STANDARD_REQUIRED ON
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON)
  endforeach()
endif()

if (${NCCL_COMM})
  foreach(X ${MAIN_TESTALLREDUCE_NCCL} ${MAIN_TESTMAXNORM} ${MAIN_TESTSCHULZ} ${MAIN_TESTSCHULZSINGLE} ${MAIN_TESTORTHO} ${MAIN_TESTORTHO_RESCALE})
    target_link_libraries(${X} PRIVATE ${NCCL_LIBRARY})
  endforeach()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA} ${MAIN_TESTMAXNORM} ${MAIN_TESTSCHULZ} ${MAIN_TESTSCHULZSINGLE} ${MAIN_TESTORTHO} ${MAIN_TESTORTHO_RESCALE} ${MAIN_TESTMATMUL})
    target_link_libraries(${X} PUBLIC OpenMP::OpenMP_CXX)
  endforeach()
endif()

add_test(NAME testMPI
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
         ${CMAKE_CURRENT_BINARY_DIR}/testMPI)

if (${NCCL_COMM})
  add_test(NAME testMatMul
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS_MATMUL}
         ${CMAKE_CURRENT_BINARY_DIR}/testMatMul)
  add_test(NAME testAllreduce
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS_DEVICECOMM}
         ${CMAKE_CURRENT_BINARY_DIR}/testAllreduce)
  add_test(NAME testAllreduce_nccl
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 6 ${MPIEXEC_PREFLAGS_DEVICECOMM}
         ${CMAKE_CURRENT_BINARY_DIR}/testAllreduce_nccl)
endif()

add_test(NAME testMAGMA
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testMAGMA 10)

add_test(NAME testMaxNormReplicated
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testMaxNormReplicated)

add_test(NAME testSchulz
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testSchulz)

add_test(NAME testSchulzSingle
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testSchulzSingle)

add_test(NAME testOrtho
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testOrtho)

add_test(NAME testOrthoRescale
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testOrthoRescale)

add_test(NAME testOrthoGaussian
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoGaussian.cfg)

add_test(NAME testOrthoGaussianDelta
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoGaussianDelta.cfg)

add_test(NAME testOrthoHatFunction
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoHatFunction.cfg)

add_test(NAME testOrthoGaussianRescale
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoGaussianRescale.cfg)

add_test(NAME testOrthoHatFunctionRescale
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoHatFunctionRescale.cfg)

add_test(NAME testOrthoGaussianCholesky
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoGaussianCholesky.cfg)

add_test(NAME testOrthoHatFunctionCholesky
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrtho.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/main
         ${CMAKE_CURRENT_SOURCE_DIR}/testOrthoHatFunctionCholesky.cfg)

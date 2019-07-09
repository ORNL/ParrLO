set(SOURCES_TESTMPI ${CMAKE_SOURCE_DIR}/tests/testMPI.cc)
set(SOURCES_TESTMAGMA ${CMAKE_SOURCE_DIR}/tests/testMAGMA.cc)

set(MAIN_TESTMPI testMPI)
add_executable(${MAIN_TESTMPI} ${SOURCES_TESTMPI})
set(MAIN_TESTMAGMA testMAGMA)
add_executable(${MAIN_TESTMAGMA} ${SOURCES_TESTMAGMA})

foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA})
  target_link_libraries(${X} PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(${X} PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(${X} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endforeach()

if (${BUILD_FOR_GPU})
  foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA})
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
  foreach(X ${MAIN_TESTMPI} ${MAIN_TESTMAGMA})
    set_target_properties(${X} PROPERTIES
      CXX_STANDARD_REQUIRED ON
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON)
  endforeach()
endif()

add_test(NAME testMPI
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
         ${CMAKE_CURRENT_BINARY_DIR}/testMPI)

add_test(NAME testMAGMA
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testMAGMA 10)

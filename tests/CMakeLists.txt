add_executable(testMPI
               ${CMAKE_SOURCE_DIR}/tests/testMPI.cc)
target_link_libraries(testMPI ${LINK_LIBRARIES})
add_test(NAME testMPI
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
                 ${CMAKE_CURRENT_BINARY_DIR}/testMPI)


set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cc
   )

add_executable(main ${SOURCES})
target_link_libraries(main PRIVATE ${BLAS_LIBRARIES})
target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${MPI_CXX_LIBRARIES})
if (${BUILD_FOR_GPU})
  target_link_libraries(main PRIVATE PkgConfig::MAGMA)
  set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD_REQUIRED ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
else()
  set_target_properties(main PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
endif()

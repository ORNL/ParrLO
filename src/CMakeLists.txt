set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cc
            ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Matrix_def.cpp
            ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Replicated_def.cpp
            ${CMAKE_SOURCE_DIR}/src/MatrixClasses/Timer.cpp)

set(MAIN main)
add_executable(${MAIN} ${SOURCES})

foreach(X ${MAIN})
  target_link_libraries(${X} PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(${X} PRIVATE ${MPI_CXX_LIBRARIES})
  target_link_libraries(${X} PRIVATE Boost::program_options Boost::boost)
  target_include_directories(${X} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endforeach()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  foreach(X ${MAIN})
    target_link_libraries(${X} PUBLIC OpenMP::OpenMP_CXX)
  endforeach()
endif()

if (${BUILD_FOR_GPU})
  foreach(X ${MAIN})
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
  foreach(X ${MAIN})
    set_target_properties(${X} PROPERTIES
      CXX_STANDARD_REQUIRED ON
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON)
  endforeach()
endif()

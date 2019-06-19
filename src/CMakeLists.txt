set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cc)
set(SOURCES_MATRIX ${CMAKE_SOURCE_DIR}/src/matrix_example.cc)
set(SOURCES_TUTORIAL ${CMAKE_SOURCE_DIR}/src/tutorial4_copy.cc)

add_executable(main ${SOURCES})
set(MAIN_MATRIX matrix_example)
add_executable(${MAIN_MATRIX} ${SOURCES_MATRIX})
set(MAIN_TUTORIAL main_tutorial)
add_executable(${MAIN_TUTORIAL} ${SOURCES_TUTORIAL})

foreach(X main ${MAIN_MATRIX} ${MAIN_TUTORIAL})
  target_link_libraries(${X} PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(${X} PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(${X} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endforeach()

if (${BUILD_FOR_GPU})
  foreach(X main ${MAIN_MATRIX} ${MAIN_TUTORIAL})
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
  foreach(X main ${MAIN_MATRIX} ${MAIN_TUTORIAL})
    set_target_properties(${X} PROPERTIES
      CXX_STANDARD_REQUIRED ON
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED ON)
  endforeach()
endif()

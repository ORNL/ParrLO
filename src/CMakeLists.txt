add_definitions(-DUSE_MAGMA)
set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cc
   )

set(MAIN main)
add_executable(${MAIN} ${SOURCES})

target_link_libraries(${MAIN} ${LINK_LIBRARIES}
                     )

set(SOURCES_MATRIX ${CMAKE_SOURCE_DIR}/src/matrix_example.cc)

set(MAIN_MATRIX matrix_example)
add_executable(${MAIN_MATRIX} ${SOURCES_MATRIX})
target_link_libraries(${MAIN_MATRIX} ${LINK_LIBRARIES}
                     )


set(SOURCES_TUTORIAL ${CMAKE_SOURCE_DIR}/src/tutorial4_copy.cc
   )

set(MAIN_TUTORIAL main_tutorial)
add_executable(${MAIN_TUTORIAL} ${SOURCES_TUTORIAL})
target_link_libraries(${MAIN_TUTORIAL} ${LINK_LIBRARIES}
                     )

set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cc
   )

set(MAIN main)
add_executable(${MAIN} ${SOURCES})

target_link_libraries(${MAIN} ${LINK_LIBRARIES}
                     )



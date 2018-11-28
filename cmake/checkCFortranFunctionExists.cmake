#note: adapted from BML
macro(check_C_Fortran_function_exists FUNCTION VARIABLE)
  # Test various naming schemes for calling a Fortran function from C.
  set(${VARIABLE} ${FUNCTION}-NOTFOUND)
  foreach(FUNC ${FUNCTION} ${FUNCTION}_ ${FUNCTION}__)
    check_function_exists(${FUNC} HAVE_${FUNC})
    if(HAVE_${FUNC})
      set(${VARIABLE} ${FUNC})
      break()
    endif()
  endforeach()
  if(${ARGC} GREATER 2)
    if(${ARGV2} STREQUAL REQUIRED AND NOT ${VARIABLE})
      message(FATAL_ERROR "Can not find function ${FUNCTION}")
    endif()
  endif()
endmacro()


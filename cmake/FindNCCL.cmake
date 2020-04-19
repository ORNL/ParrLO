# module for NCCL
include(FindPackageHandleStandardArgs)

IF(NOT NCCL_DIR)
  SET(NCCL_DIR "$ENV{CUDA_DIR}")
ENDIF()

find_path(
  NCCL_INCLUDE_DIR nccl.h
  HINTS ${NCCL_DIR}
  PATH_SUFFIXES include
)

find_library(
  NCCL_LIBRARY
  NAMES nccl
  HINTS ${NCCL_DIR}
  PATH_SUFFIXES lib64
)

find_package_handle_standard_args(
  NCCL DEFAULT_MSG
  NCCL_LIBRARY NCCL_INCLUDE_DIR
)

#ifndef INCLUDE_WARMUP_H
#define INCLUDE_WARMUP_H

#include <mpi.h>

#ifdef NCCL_COMM
#include "nccl.h"
#endif

void warmup_MPI_pt2pt(MPI_Comm comm);

#ifdef NCCL_COMM
void warmup_NCCL(ncclComm_t);
#endif

#endif

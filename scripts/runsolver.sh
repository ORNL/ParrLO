#!/bin/bash

export ENABLE_SHARP="-E HCOLL_ENABLE_SHARP=2"

export ENABLE_HCOLL="-mca coll_hcoll_enable 1"

NUMNODES=16

NUMRSRC=$((NUMNODES*6))

set PAMI_DISABLE_IPC = 1

jsrun -n $NUMRSRC -a 1 -c 1 -g 1 -r 6 $ENABLE_SHARP --smpiargs="-gpu $ENABLE_HCOLL" -d cyclic main -c ../../input/input.cfg

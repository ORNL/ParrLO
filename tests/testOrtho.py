#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test Ortho...")

nargs=len(sys.argv)

mpicmd = sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
for i in range(4,nargs-2):
  mpicmd = mpicmd + " "+sys.argv[i]
print("MPI run command: {}".format(mpicmd))

exe = sys.argv[nargs-2]
inp = sys.argv[nargs-1]
print("Input file: %s"%inp)

#run main code
command = "{} {} -c {}".format(mpicmd,exe,inp)
output = subprocess.check_output(command,shell=True)

#analyse standard output
lines=output.split(b'\n')

tol = 1.e-8
for line in lines:
  #check orthogonality before orthogonalization
  if line.count(b'orthogonality') and line.count(b'before'):
    words=line.split()
    print(words)
    delta = eval(words[5])
    print("Departure from orthogonality before orthogonalization = {}".format(delta))
    if delta<100.*tol:
      print("Departure from orthogonality before orthogonalization too small: {}".format(delta))
      sys.exit(1)

  #check orthogonality after orthogonalization
  if line.count(b'orthogonality') and line.count(b'after'):
    words=line.split()
    delta = eval(words[5])
    if delta>tol:
      print("TEST FAILED: Orthogonalization not achieved!")
      print("Departure from orthogonality: {}".format(delta))
      sys.exit(1)

sys.exit(0)

#!/bin/bash

python ../stringify.py sap.cl sapCL >sapKernels.h
python ../stringify.py broadphaseKernel.cl broadphaseKernelCL > broadphaseKernel.h



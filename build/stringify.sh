#!/bin/sh
./premake4_osx --file=stringifyKernel.lua --kernelfile="../opencl/gpu_rigidbody_pipeline2/satClipHullContacts.cl" --headerfile="../opencl/gpu_rigidbody_pipeline2/satClipKernels.h" --stringname="satClipKernelsCL" stringify


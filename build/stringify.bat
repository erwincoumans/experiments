
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/gpu_rigidbody_pipeline2/sat.cl" --headerfile="../opencl/gpu_rigidbody_pipeline2/satKernels.h" --stringname="satKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/gpu_rigidbody_pipeline2/satClipHullContacts.cl" --headerfile="../opencl/gpu_rigidbody_pipeline2/satClipKernels.h" --stringname="satClipKernelsCL" stringify
pause

@echo off

premake4 --file=stringifyKernel.lua --kernelfile="../dynamics/basic_demo/Stubs/batchingKernels.cl" --headerfile="../dynamics/basic_demo/Stubs/batchingKernels.h" --stringname="batchingKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../dynamics/basic_demo/Stubs/ChNarrowphaseKernels.cl" --headerfile="../dynamics/basic_demo/Stubs/ChNarrowphaseKernels.h" --stringname="narrowphaseKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../dynamics/basic_demo/Stubs/SolverKernels.cl" --headerfile="../dynamics/basic_demo/Stubs/SolverKernels.h" --stringname="solverKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/vector_add/VectorAddKernels.cl" --headerfile="../opencl/vector_add/VectorAddKernels.h" --stringname="vectorAddCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/broadphase_benchmark/broadphaseKernel.cl" --headerfile="../opencl/broadphase_benchmark/broadphaseKernel.h" --stringname="broadphaseKernelCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/broadphase_benchmark/sap.cl" --headerfile="../opencl/broadphase_benchmark/sapKernels.h" --stringname="sapCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/broadphase_benchmark/sapFast.cl" --headerfile="../opencl/broadphase_benchmark/sapFastKernels.h" --stringname="sapFastCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/gpu_rigidbody_pipeline2/sat.cl" --headerfile="../opencl/gpu_rigidbody_pipeline2/satKernels.h" --stringname="satKernelsCL" stringify

premake4 --file=stringifyKernel.lua --kernelfile="../opencl/gpu_rigidbody_pipeline2/satClipHullContacts.cl" --headerfile="../opencl/gpu_rigidbody_pipeline2/satClipKernels.h" --stringname="satClipKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/global_atomics/global_atomics.cl" --headerfile="../opencl/global_atomics/globalAtomicsKernel.h" --stringname="globalAtomicsKernelString" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Sort/RadixSortStandardKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Sort/RadixSortStandardKernelsCL.h" --stringname="radixSortStandardKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Sort/RadixSortSimpleKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Sort/RadixSortSimpleKernelsCL.h" --stringname="radixSortSimpleKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Sort/RadixSort32Kernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Sort/RadixSort32KernelsCL.h" --stringname="radixSort32KernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Search/BoundSearchKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Search/BoundSearchKernelsCL.h" --stringname="boundSearchKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Scan/PrefixScanKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Scan/PrefixScanKernelsCL.h" --stringname="prefixScanKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Fill/FillKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Fill/FillKernelsCL.h" --stringname="fillKernelsCL" stringify
premake4 --file=stringifyKernel.lua --kernelfile="../opencl/primitives/AdlPrimitives/Copy/CopyKernels.cl" --headerfile="../opencl/primitives/AdlPrimitives/Copy/CopyKernelsCL.h" --stringname="copyKernelsCL" stringify


pause
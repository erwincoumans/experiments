/*
		2011 Takahiro Harada
*/


//ADL_ENABLE_CL and ADL_ENABLE_DX11 can be set in the build system using C/C++ preprocessor defines
//#define ADL_ENABLE_CL
//#define ADL_ENABLE_DX11

//#define ADL_CL_FORCE_UNCACHE_KERNEL
#define ADL_CL_DUMP_MEMORY_LOG

//load the kernels from string instead of loading them from file
#define ADL_LOAD_KERNEL_FROM_STRING
#define ADL_DUMP_DX11_ERROR

#include "btFillCL.h"
#include "../basic_initialize/btOpenCLUtils.h"
#include "btBufferInfoCL.h"
#include "btLauncherCL.h"

#ifdef _WIN32
#define FILL_CL_PROGRAM_PATH "..\\..\\opencl\\primitives\\AdlPrimitives\\Fill\\FillKernels.cl"
#else
#define FILL_CL_PROGRAM_PATH "../opencl/primitives/AdlPrimitives/Fill/FillKernels.cl"
#endif

btFillCL::btFillCL(cl_context ctx, cl_device_id device, cl_command_queue queue)
:m_commandQueue(queue)
{
	char* kernelSource = 0;
	cl_int pErrNum;
	const char* additionalMacros = "";

	cl_program fillProg = btOpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, FILL_CL_PROGRAM_PATH);
	btAssert(fillProg);

	m_fillIntKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillIntKernel", &pErrNum, fillProg,additionalMacros );
	btAssert(m_fillIntKernel);

	m_fillUnsignedIntKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillUnsignedIntKernel", &pErrNum, fillProg,additionalMacros );
	btAssert(m_fillIntKernel);

	

	m_fillKernelInt2 = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "FillInt2Kernel", &pErrNum, fillProg,additionalMacros );
	btAssert(m_fillKernelInt2);
	
}

btFillCL::~btFillCL()
{
	clReleaseKernel(m_fillKernelInt2);
	clReleaseKernel(m_fillIntKernel);
	clReleaseKernel(m_fillUnsignedIntKernel);

}
		

void btFillCL::execute(btOpenCLArray<int>& src, const int& value, int n, int offset)
{
	btAssert( n>0 );
	btConstData constBuffer;
	{
		constBuffer.m_offset = offset;
		constBuffer.m_n = n;
		constBuffer.m_UnsignedData = btMakeUnsignedInt4( value,value,value,value );
	}

	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( src.getBufferCL() ) };

		btLauncherCL launcher( m_commandQueue, m_fillUnsignedIntKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst( constBuffer );
		launcher.launch1D( n );
	}
}


void btFillCL::execute(btOpenCLArray<unsigned int>& src, const unsigned int& value, int n, int offset)
{
	btAssert( n>0 );

	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( src.getBufferCL() ) };

		btLauncherCL launcher( m_commandQueue, m_fillUnsignedIntKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst( n );
        launcher.setConst(value);
		launcher.launch1D( n );
	}
}

void btFillCL::execute(btOpenCLArray<btInt2> &src, const btInt2 &value, int n, int offset)
{
	btAssert( n>0 );
	btConstData constBuffer;
	{
		constBuffer.m_offset = offset;
		constBuffer.m_n = n;
		constBuffer.m_data = btMakeInt4( value.x, value.y, 0, 0 );
	}

	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( src.getBufferCL() ) };

		btLauncherCL launcher(m_commandQueue, m_fillKernelInt2);
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst(n);
		launcher.setConst(value);
		launcher.setConst(offset);

		//( constBuffer );
		launcher.launch1D( n );
	}
}

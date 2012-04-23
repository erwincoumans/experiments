
#ifndef BT_LAUNCHER_CL_H
#define BT_LAUNCHER_CL_H

#include "btBufferInfoCL.h"
#include "LinearMath/btMinMax.h"

class btLauncherCL
{

	cl_command_queue m_commandQueue;
	cl_kernel m_kernel;
	int m_idx;


	public:

		btLauncherCL(cl_command_queue queue, cl_kernel kernel)
			:m_commandQueue(queue),
			m_kernel(kernel),
			m_idx(0)
		{
		}

		inline void setBuffers( btBufferInfoCL* buffInfo, int n )
		{
			for(int i=0; i<n; i++)
			{
				cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &buffInfo[i].m_clBuffer);
				btAssert( status == CL_SUCCESS );
			}
		}
		template<typename T>
		inline void setConst( const T& consts )
		{
			
			int sz=sizeof(T);
			cl_int status = clSetKernelArg( m_kernel, m_idx++, sz, &consts );
			btAssert( status == CL_SUCCESS );
		}

		inline void launch1D( int numThreads, int localSize = 64)
		{
			launch2D( numThreads, 1, localSize, 1 );
		}

		inline void launch2D( int numThreadsX, int numThreadsY, int localSizeX, int localSizeY )
		{
			size_t gRange[3] = {1,1,1};
			size_t lRange[3] = {1,1,1};
			lRange[0] = localSizeX;
			lRange[1] = localSizeY;
			gRange[0] = btMax((size_t)1, (numThreadsX/lRange[0])+(!(numThreadsX%lRange[0])?0:1));
			gRange[0] *= lRange[0];
			gRange[1] = btMax((size_t)1, (numThreadsY/lRange[1])+(!(numThreadsY%lRange[1])?0:1));
			gRange[1] *= lRange[1];

			cl_int status = clEnqueueNDRangeKernel( m_commandQueue, 
				m_kernel, 2, NULL, gRange, lRange, 0,0,0 );
			btAssert( status == CL_SUCCESS );

		}
};



#endif //BT_LAUNCHER_CL_H

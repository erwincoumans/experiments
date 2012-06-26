
#ifndef BT_LAUNCHER_CL_H
#define BT_LAUNCHER_CL_H

#include "btBufferInfoCL.h"
#include "LinearMath/btMinMax.h"
#include "../broadphase_benchmark/btOpenCLArray.h"

struct btKernelArgData
{
    int m_isBuffer;
    int m_argIndex;
    int m_argSizeInBytes;
    union
    {
        cl_mem m_clBuffer;
        unsigned char m_argData[16];
    };
    
};

class btLauncherCL
{

	cl_command_queue m_commandQueue;
	cl_kernel m_kernel;
	int m_idx;

    btAlignedObjectArray<btKernelArgData> m_kernelArguments;
   
    
    int m_serializationSizeInBytes;

	public:

     btAlignedObjectArray<btOpenCLArray<unsigned char>* > m_arrays;
    
		btLauncherCL(cl_command_queue queue, cl_kernel kernel)
			:m_commandQueue(queue),
			m_kernel(kernel),
			m_idx(0)
		{
            m_serializationSizeInBytes = sizeof(int);
		}
    
        virtual ~btLauncherCL()
        {
            for (int i=0;i<m_arrays.size();i++)
            {
                clReleaseMemObject(m_arrays[i]->getBufferCL());
            }
        }

		inline void setBuffers( btBufferInfoCL* buffInfo, int n )
		{
			for(int i=0; i<n; i++)
			{
                btKernelArgData kernelArg;
                kernelArg.m_argIndex = m_idx;
                kernelArg.m_isBuffer = 1;
                kernelArg.m_clBuffer = buffInfo[i].m_clBuffer;
            
                cl_mem_info param_name = CL_MEM_SIZE;
                size_t param_value;
                size_t sizeInBytes = sizeof(size_t);
                size_t actualSizeInBytes;
                cl_int err;
                err = clGetMemObjectInfo (	kernelArg.m_clBuffer,
                                          param_name,
                                          sizeInBytes,
                                          &param_value,
                                          &actualSizeInBytes);
                
                btAssert( err == CL_SUCCESS );
                kernelArg.m_argSizeInBytes = param_value;
                
                m_kernelArguments.push_back(kernelArg);
                m_serializationSizeInBytes+= sizeof(btKernelArgData);
                m_serializationSizeInBytes+=param_value;
                
                cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &buffInfo[i].m_clBuffer);
				btAssert( status == CL_SUCCESS );
            }
		}
    
    int getSerializationBufferSize() const 
    {
        return m_serializationSizeInBytes;
    }
    
    inline int deserializeArgs(unsigned char* buf, int bufSize, cl_context ctx)
    {
        int index=0;
        
        int numArguments = *(int*) &buf[index];
        index+=sizeof(int);
        
        for (int i=0;i<numArguments;i++)
        {
            btKernelArgData* arg = (btKernelArgData*)&buf[index];
            index+=sizeof(btKernelArgData);
            if (arg->m_isBuffer)
            {
                btOpenCLArray<unsigned char>* clData = new btOpenCLArray<unsigned char>(ctx,m_commandQueue, arg->m_argSizeInBytes);
                clData->resize(arg->m_argSizeInBytes);
                
                clData->copyFromHostPointer(&buf[index], arg->m_argSizeInBytes);
                
                arg->m_clBuffer = clData->getBufferCL();
                
                m_arrays.push_back(clData);
                
                cl_int status = clSetKernelArg( m_kernel, m_idx++, sizeof(cl_mem), &arg->m_clBuffer);
				btAssert( status == CL_SUCCESS );
                index+=arg->m_argSizeInBytes;
            } else 
            {
                cl_int status = clSetKernelArg( m_kernel, m_idx++, arg->m_argSizeInBytes, &arg->m_argData);
				btAssert( status == CL_SUCCESS );
            }
        }
        return index;
    }
    inline int serializeArguments(unsigned char* destBuffer, int destBufferCapacity)
    {
        assert(destBufferCapacity>=m_serializationSizeInBytes);
        
        //todo: use the btSerializer for this to allow for 32/64bit, endianness etc        
        int numArguments = m_kernelArguments.size();
        int curBufferSize = 0;
        int* dest = (int*)&destBuffer[curBufferSize];
        *dest = numArguments;
        curBufferSize += sizeof(int);
        
        
        
        for (int i=0;i<this->m_kernelArguments.size();i++)
        {
            btKernelArgData* arg = (btKernelArgData*) &destBuffer[curBufferSize];
            *arg = m_kernelArguments[i];
            curBufferSize+=sizeof(btKernelArgData);
            if (arg->m_isBuffer==1)
            {
                //copy the OpenCL buffer content
                cl_int status = 0;
                status = clEnqueueReadBuffer( m_commandQueue, arg->m_clBuffer, 0, 0, arg->m_argSizeInBytes,
                                             &destBuffer[curBufferSize], 0,0,0 );
                btAssert( status==CL_SUCCESS );
                clFinish(m_commandQueue);
                curBufferSize+=arg->m_argSizeInBytes;
            }
            
        }
        return curBufferSize;
        
    
    }
		template<typename T>
		inline void setConst( const T& consts )
		{
			int sz=sizeof(T);
            btKernelArgData kernelArg;
            kernelArg.m_argIndex = m_idx;
            kernelArg.m_isBuffer = 0;
            T* destArg = (T*)kernelArg.m_argData;
            *destArg = consts;
            kernelArg.m_argSizeInBytes = sizeof(T);
            m_kernelArguments.push_back(kernelArg);
            m_serializationSizeInBytes+=sizeof(btKernelArgData);
            
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
            if (status != CL_SUCCESS)
            {
                printf("Error: OpenCL status = %d\n",status);
            }
			btAssert( status == CL_SUCCESS );

		}
};



#endif //BT_LAUNCHER_CL_H

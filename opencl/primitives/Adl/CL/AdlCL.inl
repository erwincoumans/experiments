/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Author Takahiro Harada

#pragma comment(lib,"OpenCL.lib")
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>


struct DeviceCL : public Device
{
	typedef DeviceUtils::Config Config;


	__inline
	DeviceCL() : Device( TYPE_CL ){}
	__inline
	void* getContext() const { return m_context; }
	__inline
	void initialize(const Config& cfg);
	__inline
	void release();

	template<typename T>
	__inline
	void allocate(Buffer<T>* buf, int nElems, BufferBase::BufferType type);

	template<typename T>
	__inline
	void deallocate(Buffer<T>* buf);

	template<typename T>
	__inline
	void copy(Buffer<T>* dst, const Buffer<T>* src, int nElems, int offsetNElems = 0);

	template<typename T>
	__inline
	void copy(T* dst, const Buffer<T>* src, int nElems, int srcOffsetNElems = 0);

	template<typename T>
	__inline
	void copy(Buffer<T>* dst, const T* src, int nElems, int dstOffsetNElems = 0);

	__inline
	void waitForCompletion() const;

	__inline
	void getDeviceName( char nameOut[128] ) const;

	__inline
	static
	int getNDevices();



	enum
	{
		MAX_NUM_DEVICES = 2,
	};
	
	cl_context m_context;
	cl_command_queue m_commandQueue;

	cl_device_id m_deviceIdx;
};

//===
//===

void DeviceCL::initialize(const Config& cfg)
{
//	DeviceUtils::create( cfg, (DeviceCL*)this );
	{
//		dd = new DeviceCL();

		DeviceCL* deviceData = (DeviceCL*)this;

//		cl_device_type deviceType = (driverType == DRIVER_HARDWARE)? CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU;
		cl_device_type deviceType = (cfg.m_type== Config::DEVICE_GPU)? CL_DEVICE_TYPE_GPU: CL_DEVICE_TYPE_CPU;
//		int numContextQueuePairsToCreate = 1;
		bool enableProfiling = false;
#ifdef _DEBUG
		enableProfiling = true;
#endif
		cl_int status;

		cl_platform_id platform;
		{
			cl_uint nPlatforms = 0;
			status = clGetPlatformIDs(0, NULL, &nPlatforms);
			ADLASSERT( status == CL_SUCCESS );

			cl_platform_id pIdx[5];
			status = clGetPlatformIDs(nPlatforms, pIdx, NULL);
			ADLASSERT( status == CL_SUCCESS );

			cl_uint atiIdx = -1;
			cl_uint intelIdx = -1;
			cl_uint nvIdx = -1;

			for(cl_uint i=0; i<nPlatforms; i++)
			{
				char buff[512];
				status = clGetPlatformInfo( pIdx[i], CL_PLATFORM_VENDOR, 512, buff, 0 );
				ADLASSERT( status == CL_SUCCESS );

				if( strcmp( buff, "NVIDIA Corporation" )==0 ) nvIdx = i;
				if( strcmp( buff, "Advanced Micro Devices, Inc." )==0 ) atiIdx = i;
				if( strcmp( buff, "Intel Corporation" )==0 ) intelIdx = i;
			}

			if( deviceType == CL_DEVICE_TYPE_GPU )
			{
				switch( cfg.m_vendor )
				{
				case DeviceUtils::Config::VD_AMD:
					ADLASSERT(atiIdx != -1 );
					platform = pIdx[atiIdx];
					break;
				case DeviceUtils::Config::VD_NV:
					ADLASSERT(nvIdx != -1 );
					platform = pIdx[nvIdx];
					break;
				default:
					ADLASSERT(0);
					break;
				};
			}
			else if( deviceType == CL_DEVICE_TYPE_CPU )
			{
				switch( cfg.m_vendor )
				{
				case DeviceUtils::Config::VD_AMD:
					ADLASSERT(atiIdx != -1 );
					platform = pIdx[atiIdx];
					break;
				case DeviceUtils::Config::VD_INTEL:
					ADLASSERT(intelIdx != -1 );
					platform = pIdx[intelIdx];
					break;
				default:
					ADLASSERT(0);
					break;
				};
			}
		}

		cl_uint numDevice;
		status = clGetDeviceIDs( platform, deviceType, 0, NULL, &numDevice );

//		ADLASSERT( cfg.m_deviceIdx < (int)numDevice );

		adlDebugPrintf("CL: %d %s Devices ", numDevice, (deviceType==CL_DEVICE_TYPE_GPU)? "GPU":"CPU");

//		numContextQueuePairsToCreate = min( (int)numDevice, numContextQueuePairsToCreate );
//		numContextQueuePairsToCreate = ( (int)numDevice < numContextQueuePairsToCreate )? numDevice : numContextQueuePairsToCreate;
		
		cl_device_id deviceIds[ MAX_NUM_DEVICES ];

		status = clGetDeviceIDs( platform, deviceType, numDevice, deviceIds, NULL );
		ADLASSERT( status == CL_SUCCESS );

		{	int i = min( (int)numDevice-1, cfg.m_deviceIdx );
			m_deviceIdx = deviceIds[i];
			deviceData->m_context = clCreateContext( NULL, 1, &deviceData->m_deviceIdx, NULL, NULL, &status );
			ADLASSERT( status == CL_SUCCESS );

			char buff[512];
			status = clGetDeviceInfo( deviceData->m_deviceIdx, CL_DEVICE_NAME, sizeof(buff), &buff, NULL );
			ADLASSERT( status == CL_SUCCESS );

			adlDebugPrintf("[%s]\n", buff);

			deviceData->m_commandQueue = clCreateCommandQueue( deviceData->m_context, deviceData->m_deviceIdx, (enableProfiling)?CL_QUEUE_PROFILING_ENABLE:NULL, NULL );

			ADLASSERT( status == CL_SUCCESS );

		//	status = clSetCommandQueueProperty( commandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, 0 );
		//	CLASSERT( status == CL_SUCCESS );

			if(0)
			{
				cl_bool image_support;
				clGetDeviceInfo(deviceData->m_deviceIdx, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
				adlDebugPrintf("	CL_DEVICE_IMAGE_SUPPORT : %s\n", image_support?"Yes":"No");
			}
		}

	}
}

void DeviceCL::release()
{
	clReleaseCommandQueue( m_commandQueue );
	clReleaseContext( m_context );
}

template<typename T>
void DeviceCL::allocate(Buffer<T>* buf, int nElems, BufferBase::BufferType type)
{
	buf->m_device = this;
	buf->m_size = nElems;
	buf->m_ptr = 0;

	if( type == BufferBase::BUFFER_CONST ) return;
	cl_int status = 0;
	if( type == BufferBase::BUFFER_ZERO_COPY )
		buf->m_ptr = (T*)clCreateBuffer( m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(T)*nElems, 0, &status );
	else if( type == BufferBase::BUFFER_RAW )
		buf->m_ptr = (T*)clCreateBuffer( m_context, CL_MEM_WRITE_ONLY, sizeof(T)*nElems, 0, &status );
	else
		buf->m_ptr = (T*)clCreateBuffer( m_context, CL_MEM_READ_WRITE, sizeof(T)*nElems, 0, &status );
	ADLASSERT( status == CL_SUCCESS );
}

template<typename T>
void DeviceCL::deallocate(Buffer<T>* buf)
{
	if( buf->m_ptr ) clReleaseMemObject( (cl_mem)buf->m_ptr );
	buf->m_device = 0;
	buf->m_size = 0;
	buf->m_ptr = 0;
}

template<typename T>
void DeviceCL::copy(Buffer<T>* dst, const Buffer<T>* src, int nElems, int offsetNElems )
{
	if( dst.m_device->m_type == TYPE_CL || src.m_device->m_type == TYPE_HOST )
	{
		copy( dst, src.m_ptr, nElems, offsetNElems );
	}
	else if( src.m_device->m_type == TYPE_CL || dst.m_device->m_type == TYPE_HOST )
	{
		copy( dst->m_ptr, src, nElems, offsetNElems );
	}
	else
	{
		ADLASSERT( 0 );
	}
}

template<typename T>
void DeviceCL::copy(T* dst, const Buffer<T>* src, int nElems, int srcOffsetNElems )
{
	cl_int status = 0;
	status = clEnqueueReadBuffer( m_commandQueue, (cl_mem)src->m_ptr, 0, sizeof(T)*srcOffsetNElems, sizeof(T)*nElems,
		dst, 0,0,0 );
	ADLASSERT( status == CL_SUCCESS );
}

template<typename T>
void DeviceCL::copy(Buffer<T>* dst, const T* src, int nElems, int dstOffsetNElems )
{
	cl_int status = 0;
	status = clEnqueueWriteBuffer( m_commandQueue, (cl_mem)dst->m_ptr, 0, sizeof(T)*dstOffsetNElems, sizeof(T)*nElems,
		src, 0,0,0 );
	ADLASSERT( status == CL_SUCCESS );
}

void DeviceCL::waitForCompletion() const
{
	clFinish( m_commandQueue );
}

int DeviceCL::getNDevices()
{
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	cl_int status;

	cl_platform_id platform;
	{
		cl_uint nPlatforms = 0;
		status = clGetPlatformIDs(0, NULL, &nPlatforms);
		ADLASSERT( status == CL_SUCCESS );

		cl_platform_id pIdx[5];
		status = clGetPlatformIDs(nPlatforms, pIdx, NULL);
		ADLASSERT( status == CL_SUCCESS );

		cl_uint nvIdx = -1;
		cl_uint atiIdx = -1;
		for(cl_uint i=0; i<nPlatforms; i++)
		{
			char buff[512];
			status = clGetPlatformInfo( pIdx[i], CL_PLATFORM_VENDOR, 512, buff, 0 );
			ADLASSERT( status == CL_SUCCESS );

			if( strcmp( buff, "NVIDIA Corporation" )==0 ) nvIdx = i;
			if( strcmp( buff, "Advanced Micro Devices, Inc." )==0 ) atiIdx = i;
		}

		if( deviceType == CL_DEVICE_TYPE_GPU )
		{
			if( nvIdx != -1 ) platform = pIdx[nvIdx];
			else platform = pIdx[atiIdx];
		}
		else if( deviceType == CL_DEVICE_TYPE_CPU )
		{
			platform = pIdx[atiIdx];
		}
	}

	cl_uint numDevice;
	status = clGetDeviceIDs( platform, deviceType, 0, NULL, &numDevice );
	ADLASSERT( status == CL_SUCCESS );

	return numDevice;
}

void DeviceCL::getDeviceName( char nameOut[128] ) const
{
	cl_int status;
	status = clGetDeviceInfo( m_deviceIdx, CL_DEVICE_NAME, sizeof(char)*128, nameOut, NULL );
	ADLASSERT( status == CL_SUCCESS );
}

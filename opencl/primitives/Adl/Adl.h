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

//#pragma once
#ifndef ADL_H
#define ADL_H

#include <Adl/AdlConfig.h>
#include <Adl/AdlError.h>
#include <algorithm>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


__inline
void AdlAllocate();
__inline
void AdlDeallocate();

enum DeviceType
{
	TYPE_CL = 0,
#if defined(ADL_ENABLE_DX11)
	TYPE_DX11 = 1,
#endif
	TYPE_HOST,
};


struct Device;

struct BufferBase
{
	enum BufferType
	{
		BUFFER,

		//	for dx
		BUFFER_CONST,
		BUFFER_STAGING,
		BUFFER_APPEND,
		BUFFER_RAW,
		BUFFER_W_COUNTER,
		BUFFER_INDEX,
		BUFFER_VERTEX,

		//	for cl
		BUFFER_ZERO_COPY,

	};
};

class DeviceUtils
{
	public:
		struct Config
		{
			enum DeviceType
			{
				DEVICE_GPU,
				DEVICE_CPU,
			};

			//	for CL
			enum DeviceVendor
			{
				VD_AMD,
				VD_INTEL,
				VD_NV,
			};

			Config() : m_type(DEVICE_GPU), m_deviceIdx(0), m_vendor(VD_AMD){}

			DeviceType m_type;
			int m_deviceIdx;
			DeviceVendor m_vendor;
		};

		__inline
		static
		int getNDevices( DeviceType type );
		__inline
		static Device* allocate( DeviceType type, Config& cfg );
		__inline
		static void deallocate( Device* deviceData );
		__inline
		static void waitForCompletion( const Device* deviceData );
};

//==========================
//	DeviceData
//==========================
struct Device
{
	typedef DeviceUtils::Config Config;

	Device( DeviceType type ) : m_type( type ){}

	virtual void* getContext() const { return 0; }
	virtual void initialize(const Config& cfg){}
	virtual void release(){}
	virtual void waitForCompletion() const {}
	virtual void getDeviceName( char nameOut[128] ) const {}

	DeviceType m_type;
};

//==========================
//	Buffer
//==========================

template<typename T>
struct HostBuffer;
//	overload each deviceDatas
template<typename T>
struct Buffer : public BufferBase
{
	__inline
	Buffer();
	__inline
	Buffer(const Device* device, int nElems, BufferType type = BUFFER );
	__inline
	virtual ~Buffer();
	
	__inline
	void setRawPtr( const Device* device, T* ptr, int size, BufferType type = BUFFER );
	__inline
	void allocate(const Device* device, int nElems, BufferType type = BUFFER );
	__inline
	void write(T* hostPtr, int nElems, int dstOffsetNElems = 0);
	__inline
	void read(T* hostPtr, int nElems, int srcOffsetNElems = 0) const;
	__inline
	Buffer<T>& operator = (const HostBuffer<T>& host);
	__inline
	int getSize() { return m_size; }

	DeviceType getType() const { ADLASSERT( m_device ); return m_device->m_type; }


	const Device* m_device;
	int m_size;
	T* m_ptr;
	//	for DX11
	void* m_uav;
	void* m_srv;
	bool m_allocated;	//	todo. move this to a bit
};

//==========================
//	HostBuffer
//==========================
struct DeviceHost;

template<typename T>
struct HostBuffer : public Buffer<T>
{
	__inline
	HostBuffer():Buffer<T>(){}
	__inline
	HostBuffer(const Device* device, int nElems, BufferType type = BUFFER ) : Buffer<T>(device, nElems, type) {}
//	HostBuffer(const Device* deviceData, T* rawPtr, int nElems);


	__inline
	T& operator[](int idx);
	__inline
	const T& operator[](int idx) const;
	__inline
	T* begin() { return m_ptr; }

	__inline
	HostBuffer<T>& operator = (const Buffer<T>& device);
};

#include <Adl/AdlKernel.h>
#if defined(ADL_ENABLE_CL)
	#include <Adl/CL/AdlCL.inl>
#endif
#if defined(ADL_ENABLE_DX11)
	#include <Adl/DX11/AdlDX11.inl>
#endif

#include <Adl/Host/AdlHost.inl>
#include <Adl/Adl.inl>
#include <Adl/AdlKernel.inl>


#include <Adl/AdlStopwatch.h>
#include <Adl/CL/AdlStopwatchCL.inl>
#if defined(ADL_ENABLE_DX11)
	#include <Adl/DX11/AdlStopwatchDX11.inl>
#endif
#include <Adl/Host/AdlStopwatchHost.inl>
#include <Adl/AdlStopwatch.inl>

#endif

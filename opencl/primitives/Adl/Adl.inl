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

void AdlAllocate()
{

}

void AdlDeallocate()
{
	if( KernelManager::s_kManager )
		delete KernelManager::s_kManager;
	KernelManager::s_kManager = NULL;
}

int DeviceUtils::getNDevices( DeviceType type )
{
	switch( type )
	{
	case TYPE_CL:
		return DeviceCL::getNDevices();
#if defined(ADL_ENABLE_DX11)
	case TYPE_DX11:
		return DeviceDX11::getNDevices();
#endif
	default:
		return 1;
	};
}

Device* DeviceUtils::allocate( DeviceType type, Config& cfg )
{
	Device* deviceData;
	switch( type )
	{
	case TYPE_CL:
		deviceData = new DeviceCL();
		break;
#if defined(ADL_ENABLE_DX11)
	case TYPE_DX11:
		deviceData = new DeviceDX11();
		break;
#endif
	case TYPE_HOST:
		deviceData = new DeviceHost();
		break;
	default:
		ADLASSERT( 0 );
		break;
	};
	deviceData->initialize( cfg );
	return deviceData;
}

void DeviceUtils::deallocate( Device* deviceData )
{
	deviceData->release();
	delete deviceData;
}

void DeviceUtils::waitForCompletion( const Device* deviceData )
{
	deviceData->waitForCompletion();
}

#if defined(ADL_ENABLE_DX11)
#define SELECT_DEVICEDATA( type, func ) \
	switch( type ) \
	{ \
	case TYPE_CL: ((DeviceCL*)m_device)->func; break; \
	case TYPE_DX11: ((DeviceDX11*)m_device)->func; break; \
	case TYPE_HOST: ((DeviceHost*)m_device)->func; break; \
	default: ADLASSERT(0); break; \
	}

#define SELECT_DEVICEDATA1( deviceData, func ) \
	switch( deviceData->m_type ) \
	{ \
	case TYPE_CL: ((DeviceCL*)deviceData)->func; break; \
	case TYPE_DX11: ((DeviceDX11*)deviceData)->func; break; \
	case TYPE_HOST: ((DeviceHost*)deviceData)->func; break; \
	default: ADLASSERT(0); break; \
	}
#else
#define SELECT_DEVICEDATA( type, func ) \
	switch( type ) \
	{ \
	case TYPE_CL: ((DeviceCL*)m_device)->func; break; \
	case TYPE_HOST: ((DeviceHost*)m_device)->func; break; \
	default: ADLASSERT(0); break; \
	}

#define SELECT_DEVICEDATA1( deviceData, func ) \
	switch( deviceData->m_type ) \
	{ \
	case TYPE_CL: ((DeviceCL*)deviceData)->func; break; \
	case TYPE_HOST: ((DeviceHost*)deviceData)->func; break; \
	default: ADLASSERT(0); break; \
	}
#endif

template<typename T>
Buffer<T>::Buffer()
{
	m_device = 0;
	m_size = 0;
	m_ptr = 0;

	m_uav = 0;
	m_srv = 0;

	m_allocated = false;
}

template<typename T>
Buffer<T>::Buffer(const Device* deviceData, int nElems, BufferType type )
{
	m_device = 0;
	allocate( deviceData, nElems, type );
}

template<typename T>
Buffer<T>::~Buffer()
{
	if( m_allocated )
	{
		if( m_device )
			SELECT_DEVICEDATA( m_device->m_type, deallocate( this ) );
	}

	m_device = 0;
	m_ptr = 0;
	m_size = 0;
}

template<typename T>
void Buffer<T>::setRawPtr( const Device* device, T* ptr, int size, BufferType type )
{
	ADLASSERT( m_device == 0 );
	ADLASSERT( type == BUFFER );	//	todo. implement
	ADLASSERT( device->m_type != TYPE_DX11 );	//	todo. implement set srv, uav

	m_device = device;
	m_ptr = ptr;
	m_size = size;
}

template<typename T>
void Buffer<T>::allocate(const Device* deviceData, int nElems, BufferType type )
{
	ADLASSERT( m_device == 0 );
	m_device = deviceData;
	m_size = 0;
	m_ptr = 0;

	m_uav = 0;
	m_srv = 0;

	SELECT_DEVICEDATA( m_device->m_type, allocate( this, nElems, type ) );
	m_allocated = true;
}

template<typename T>
void Buffer<T>::write(T* hostPtr, int nElems, int offsetNElems)
{
	SELECT_DEVICEDATA( m_device->m_type, copy(this, hostPtr, nElems, offsetNElems) );
}

template<typename T>
void Buffer<T>::read(T* hostPtr, int nElems, int offsetNElems) const
{
	SELECT_DEVICEDATA( m_device->m_type, copy(hostPtr,this, nElems, offsetNElems) );
}

template<typename T>
Buffer<T>& Buffer<T>::operator=( const HostBuffer<T>& host )
{
	ADLASSERT( host.m_size <= m_size );

	SELECT_DEVICEDATA( m_device->m_type, copy(this, host.m_ptr, host.m_size ) );

	return *this;
}



template<typename T>
T& HostBuffer<T>::operator[](int idx)
{
	return m_ptr[idx];
}

template<typename T>
const T& HostBuffer<T>::operator[](int idx) const
{
	return m_ptr[idx];
}

template<typename T>
HostBuffer<T>& HostBuffer<T>::operator = ( const Buffer<T>& device )
{
	ADLASSERT( device.m_size <= m_size );

	SELECT_DEVICEDATA1( device.m_device, copy( m_ptr, &device, device.m_size ) );

	return *this;
}

#undef SELECT_DEVICEDATA


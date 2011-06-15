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

#define PATH "..\\..\\AdlPrimitives\\Search\\BoundSearchKernels"
#define KERNEL0 "SearchSortDataLowerKernel"
#define KERNEL1 "SearchSortDataUpperKernel"

#include <AdlPrimitives/Search/BoundSearchKernelsCL.h>
#include <AdlPrimitives/Search/BoundSearchKernelsDX11.h>

template<DeviceType TYPE>
typename BoundSearch<TYPE>::Data* BoundSearch<TYPE>::allocate(const Device* device)
{
	ADLASSERT( TYPE == device->m_type );

	const char* src[] = 
#if defined(ADL_LOAD_KERNEL_FROM_STRING)
		{boundSearchKernelsCL, boundSearchKernelsDX11};
#else
		{0,0};
#endif

	Data* data = new Data;

	data->m_device = device;
	data->m_lowerSortDataKernel = KernelManager::query( device, PATH, KERNEL0, 0, src[TYPE] );
	data->m_upperSortDataKernel = KernelManager::query( device, PATH, KERNEL1, 0, src[TYPE] );
	data->m_constBuffer = new Buffer<int4>( device, 1, BufferBase::BUFFER_CONST );

	return data;
}

template<DeviceType TYPE>
void BoundSearch<TYPE>::deallocate(Data* data)
{
	delete data->m_constBuffer;
	delete data;
}

template<DeviceType TYPE>
void BoundSearch<TYPE>::execute(Data* data, Buffer<SortData>& src, u32 nSrc, Buffer<u32>& dst, u32 nDst, Option option )
{
	ADLASSERT( TYPE == src.getType() );
	ADLASSERT( TYPE == dst.getType() );

	int4 constBuffer;
	constBuffer.x = nSrc;
	constBuffer.y = nDst;

	if( option == BOUND_LOWER )
	{
		BufferInfo bInfo[] = { BufferInfo( &src, true ), BufferInfo( &dst ) };

		Launcher launcher( data->m_device, data->m_lowerSortDataKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
		launcher.setConst( *data->m_constBuffer, constBuffer );
		launcher.launch1D( nSrc, 64 );
	}
	else if( option == BOUND_UPPER )
	{
		BufferInfo bInfo[] = { BufferInfo( &src, true ), BufferInfo( &dst ) };

		Launcher launcher( data->m_device, data->m_upperSortDataKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
		launcher.setConst( *data->m_constBuffer, constBuffer );
		launcher.launch1D( nSrc+1, 64 );
	}
	else
	{
		ADLASSERT( 0 );
	}
}


#undef PATH
#undef KERNEL0
#undef KERNEL1


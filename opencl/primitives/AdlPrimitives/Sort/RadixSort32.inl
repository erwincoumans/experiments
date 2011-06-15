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


#define PATH "..\\..\\AdlPrimitives\\Sort\\RadixSort32Kernels"
#define KERNEL0 "StreamCountKernel"
#define KERNEL1 "PrefixScanKernel"
#define KERNEL2 "SortAndScatterKernel"


#include <AdlPrimitives/Sort/RadixSort32KernelsDX11.h>
#include <AdlPrimitives/Sort/RadixSort32KernelsCL.h>


//	todo. Shader compiler (2010JuneSDK) doesn't allow me to place Barriers in SortAndScatterKernel... 
//	So it only works on a GPU with 64 wide SIMD. 

template<DeviceType TYPE>
typename RadixSort32<TYPE>::Data* RadixSort32<TYPE>::allocate( const Device* device, int maxSize )
{
	ADLASSERT( TYPE == device->m_type );

	const char* src[] = 
#if defined(ADL_LOAD_KERNEL_FROM_STRING)
	{radixSort32KernelsCL , radixSort32KernelsDX11 };
//	ADLASSERT(0);
#else
	{0,0};
#endif

	Data* data = new Data;
	data->m_device = device;
	data->m_maxSize = maxSize;
	data->m_streamCountKernel = KernelManager::query( device, PATH, KERNEL0, 0, src[TYPE] );
	data->m_prefixScanKernel = KernelManager::query( device, PATH, KERNEL1, 0, src[TYPE] );
	data->m_sortAndScatterKernel = KernelManager::query( device, PATH, KERNEL2, 0, src[TYPE] );

	data->m_workBuffer0 = new Buffer<u32>( device, maxSize );
	data->m_workBuffer1 = new Buffer<u32>( device , NUM_WGS*(1<<BITS_PER_PASS) );

	for(int i=0; i<32/BITS_PER_PASS; i++)
		data->m_constBuffer[i] = new Buffer<ConstData>( device, 1, BufferBase::BUFFER_CONST );

	data->m_copyData = Copy<TYPE>::allocate( device );

	return data;
}

template<DeviceType TYPE>
void RadixSort32<TYPE>::deallocate( Data* data )
{
	delete data->m_workBuffer0;
	delete data->m_workBuffer1;
	for(int i=0; i<32/BITS_PER_PASS; i++)
		delete data->m_constBuffer[i];

	Copy<TYPE>::deallocate( data->m_copyData );

	delete data;
}

template<DeviceType TYPE>
void RadixSort32<TYPE>::execute(Data* data, Buffer<u32>& inout, int n, int sortBits /* = 32 */ )
{
	ADLASSERT( n%256 == 0 );
	ADLASSERT( n <= data->m_maxSize );
	ADLASSERT( ELEMENTS_PER_WORK_ITEM == 4 );
	ADLASSERT( BITS_PER_PASS == 4 );
	ADLASSERT( WG_SIZE == 64 );
	ADLASSERT( (sortBits&0x3) == 0 );

	Buffer<u32>* src = &inout;
	Buffer<u32>* dst = data->m_workBuffer0;
	Buffer<u32>* histogramBuffer = data->m_workBuffer1;

	const int nWGs = NUM_WGS;
	ConstData cdata;
	{
		int nBlocks = (n+ELEMENTS_PER_WORK_ITEM*WG_SIZE-1)/(ELEMENTS_PER_WORK_ITEM*WG_SIZE);
		cdata.m_n = n;
		cdata.m_nWGs = nWGs;
		cdata.m_startBit = 0;
		cdata.m_nBlocksPerWG = (nBlocks + cdata.m_nWGs - 1)/cdata.m_nWGs;
	}

	for(int ib=0; ib<sortBits; ib+=4)
	{
		cdata.m_startBit = ib;
		{
			BufferInfo bInfo[] = { BufferInfo( src, true ), BufferInfo( histogramBuffer ) };
			Launcher launcher( data->m_device, data->m_streamCountKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
		{//	prefix scan group histogram
			BufferInfo bInfo[] = { BufferInfo( histogramBuffer ) };
			Launcher launcher( data->m_device, data->m_prefixScanKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( 128, 128 );
		}
		{//	local sort and distribute
			BufferInfo bInfo[] = { BufferInfo( src, true ), BufferInfo( histogramBuffer, true ), BufferInfo( dst ) };
			Launcher launcher( data->m_device, data->m_sortAndScatterKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
		swap2( src, dst );
	}

	if( src != &inout )
	{
		Copy<TYPE>::execute( data->m_copyData, (Buffer<float>&)inout, (Buffer<float>&)*src, n );
	}
}

template<DeviceType TYPE>
void RadixSort32<TYPE>::execute(Data* data, Buffer<u32>& in, Buffer<u32>& out, int n, int sortBits /* = 32 */ )
{
	ADLASSERT( n%256 == 0 );
	ADLASSERT( n <= data->m_maxSize );
	ADLASSERT( ELEMENTS_PER_WORK_ITEM == 4 );
	ADLASSERT( BITS_PER_PASS == 4 );
	ADLASSERT( WG_SIZE == 64 );
	ADLASSERT( (sortBits&0x3) == 0 );

	Buffer<u32>* src = &in;
	Buffer<u32>* dst = data->m_workBuffer0;
	Buffer<u32>* histogramBuffer = data->m_workBuffer1;

	const int nWGs = NUM_WGS;
	ConstData cdata;
	{
		int nBlocks = (n+ELEMENTS_PER_WORK_ITEM*WG_SIZE-1)/(ELEMENTS_PER_WORK_ITEM*WG_SIZE);
		cdata.m_n = n;
		cdata.m_nWGs = nWGs;
		cdata.m_startBit = 0;
		cdata.m_nBlocksPerWG = (nBlocks + cdata.m_nWGs - 1)/cdata.m_nWGs;
	}

	if( sortBits == 4 ) dst = &out;

	for(int ib=0; ib<sortBits; ib+=4)
	{
		if( ib==4 )
		{
			dst = &out;
		}

		cdata.m_startBit = ib;
		{
			BufferInfo bInfo[] = { BufferInfo( src, true ), BufferInfo( histogramBuffer ) };
			Launcher launcher( data->m_device, data->m_streamCountKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
		{//	prefix scan group histogram
			BufferInfo bInfo[] = { BufferInfo( histogramBuffer ) };
			Launcher launcher( data->m_device, data->m_prefixScanKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( 128, 128 );
		}
		{//	local sort and distribute
			BufferInfo bInfo[] = { BufferInfo( src, true ), BufferInfo( histogramBuffer, true ), BufferInfo( dst ) };
			Launcher launcher( data->m_device, data->m_sortAndScatterKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(Launcher::BufferInfo) );
			launcher.setConst( *data->m_constBuffer[ib/4], cdata );
			launcher.launch1D( nWGs*WG_SIZE, WG_SIZE );
		}
		swap2( src, dst );
	}

	if( src != &out )
	{
		Copy<TYPE>::execute( data->m_copyData, (Buffer<float>&)out, (Buffer<float>&)*src, n );
	}
}



#undef PATH
#undef KERNEL0
#undef KERNEL1
#undef KERNEL2



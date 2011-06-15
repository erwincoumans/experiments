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

#include <AdlPrimitives/Sort/RadixSortSimple.inl>
#include <AdlPrimitives/Sort/RadixSortStandard.inl>
#include <AdlPrimitives/Sort/RadixSortAdvanced.inl>


#define DISPATCH_IMPL(x) \
	switch( data->m_option ) \
	{ \
		case SORT_SIMPLE: RadixSortSimple<TYPE>::x; break; \
		case SORT_STANDARD: RadixSortStandard<TYPE>::x; break; \
		case SORT_ADVANCED: RadixSortAdvanced<TYPE>::x; break; \
		default:ADLASSERT(0);break; \
	}

template<DeviceType TYPE>
typename RadixSort<TYPE>::Data* RadixSort<TYPE>::allocate(const Device* deviceData, int maxSize, Option option)
{
	ADLASSERT( TYPE == deviceData->m_type );

	void* dataOut;
	switch( option )
	{
	case SORT_SIMPLE:
		dataOut = RadixSortSimple<TYPE>::allocate( deviceData, maxSize, option );
		break;
	case SORT_STANDARD:
		dataOut = RadixSortStandard<TYPE>::allocate( deviceData, maxSize, option );
		break;
	case SORT_ADVANCED:
		dataOut = RadixSortAdvanced<TYPE>::allocate( deviceData, maxSize, option );
		break;
	default:
		ADLASSERT(0);
		break;
	}
	return (typename RadixSort<TYPE>::Data*)dataOut;
}

template<DeviceType TYPE>
void RadixSort<TYPE>::deallocate(Data* data)
{
	DISPATCH_IMPL( deallocate( data ) );
}

template<DeviceType TYPE>
void RadixSort<TYPE>::execute(Data* data, Buffer<SortData>& inout, int n, int sortBits)
{
	DISPATCH_IMPL( execute( data, inout, n, sortBits ) );
}


#undef DISPATCH_IMPL



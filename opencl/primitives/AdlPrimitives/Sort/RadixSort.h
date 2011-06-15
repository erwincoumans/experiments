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

#pragma once

#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>
#include <AdlPrimitives/Sort/SortData.h>
#include <AdlPrimitives/Scan/PrefixScan.h>


class RadixSortBase
{
	public:
		enum Option
		{
			SORT_SIMPLE,
			SORT_STANDARD, 
			SORT_ADVANCED
		};
};

template<DeviceType TYPE>
class RadixSort : public RadixSortBase
{
	public:
		struct Data
		{
			Option m_option;
			const Device* m_deviceData;
			typename PrefixScan<TYPE>::Data* m_scanData;
			int m_maxSize;
		};
		

		static
		Data* allocate(const Device* deviceData, int maxSize, Option option = SORT_STANDARD);

		static
		void deallocate(Data* data);

		static
		void execute(Data* data, Buffer<SortData>& inout, int n, int sortBits = 32);
};


#include <AdlPrimitives/Sort/RadixSort.inl>
#include <AdlPrimitives/Sort/RadixSortHost.inl>


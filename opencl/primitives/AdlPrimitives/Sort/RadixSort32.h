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
#include <AdlPrimitives/Copy/Copy.h>


class RadixSort32Base
{
	public:
// 		enum Option
// 		{
// 			SORT_SIMPLE,
// 			SORT_STANDARD, 
// 			SORT_ADVANCED
// 		};
};

template<DeviceType TYPE>
class RadixSort32 : public RadixSort32Base
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		enum
		{
			WG_SIZE = 64,
			ELEMENTS_PER_WORK_ITEM = 4,
			BITS_PER_PASS = 4,
			NUM_WGS = 20*6,
		};

		struct ConstData
		{
			int m_n;
			int m_nWGs;
			int m_startBit;
			int m_nBlocksPerWG;
		};

		struct Data
		{
			const Device* m_device;
			int m_maxSize;

			Kernel* m_streamCountKernel;
			Kernel* m_prefixScanKernel;
			Kernel* m_sortAndScatterKernel;

			Buffer<u32>* m_workBuffer0;
			Buffer<u32>* m_workBuffer1;
			Buffer<ConstData>* m_constBuffer[32/BITS_PER_PASS];

			typename Copy<TYPE>::Data* m_copyData;
		};

		static
		Data* allocate(const Device* device, int maxSize);

		static
		void deallocate(Data* data);

		static
		void execute(Data* data, Buffer<u32>& inout, int n, int sortBits = 32);

		static
		void execute(Data* data, Buffer<u32>& in, Buffer<u32>& out, int n, int sortBits = 32);
};


#include <AdlPrimitives/Sort/RadixSort32Host.inl>
#include <AdlPrimitives/Sort/RadixSort32.inl>

/*
		2011 Takahiro Harada
*/

#pragma once

#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>
#include <AdlPrimitives/Sort/SortData.h>
#include <AdlPrimitives/Fill/Fill.h>

namespace adl
{

class BoundSearchBase
{
	public:
		enum Option
		{
			BOUND_LOWER,
			BOUND_UPPER,
			COUNT,
		};
};

template<DeviceType TYPE>
class BoundSearch : public BoundSearchBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct Data
		{
			const Device* m_device;
			Kernel* m_lowerSortDataKernel;
			Kernel* m_upperSortDataKernel;
			Kernel* m_subtractKernel;
			Buffer<int4>* m_constBuffer;
			Buffer<u32>* m_lower;
			Buffer<u32>* m_upper;
			typename Fill<TYPE>::Data* m_fillData;
		};

		static
		Data* allocate(const Device* deviceData, int maxSize = 0);

		static
		void deallocate(Data* data);

		//	src has to be src[i].m_key <= src[i+1].m_key
		static
		void execute(Data* data, Buffer<SortData>& src, u32 nSrc, Buffer<u32>& dst, u32 nDst, Option option = BOUND_LOWER );

//		static
//		void execute(Data* data, Buffer<u32>& src, Buffer<u32>& dst, int n, Option option = );
};

#include <AdlPrimitives/Search/BoundSearchHost.inl>
#include <AdlPrimitives/Search/BoundSearch.inl>

};

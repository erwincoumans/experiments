/*
		2011 Takahiro Harada
*/

#pragma once

#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>

namespace adl
{

class FillBase
{
	public:
		enum Option
		{

		};
};

template<DeviceType TYPE>
class Fill
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct ConstData
		{
			int4 m_data;
			int m_offset;
			int m_n;
			int m_padding[2];
		};

		struct Data
		{
			const Device* m_device;
			Kernel* m_fillIntKernel;
			Kernel* m_fillInt2Kernel;
			Kernel* m_fillInt4Kernel;
			Buffer<ConstData>* m_constBuffer;
		};

		static
		Data* allocate(const Device* deviceData);

		static
		void deallocate(Data* data);

		static
		void execute(Data* data, Buffer<int>& src, const int& value, int n, int offset = 0);

		static
		void execute(Data* data, Buffer<int2>& src, const int2& value, int n, int offset = 0);

		static
		void execute(Data* data, Buffer<int4>& src, const int4& value, int n, int offset = 0);

};


#include <AdlPrimitives/Fill/FillHost.inl>
#include <AdlPrimitives/Fill/Fill.inl>

};

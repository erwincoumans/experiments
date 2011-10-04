/*
		2011 Takahiro Harada
*/

#pragma once

#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>

namespace adl
{

class CopyBase
{
	public:
		enum Option
		{
			PER_WI_1, 
			PER_WI_2, 
			PER_WI_4, 
		};
};

template<DeviceType TYPE>
class Copy : public CopyBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct Data
		{
			const Device* m_device;
			Kernel* m_copy1F4Kernel;
			Kernel* m_copy2F4Kernel;
			Kernel* m_copy4F4Kernel;
			Kernel* m_copyF1Kernel;
			Kernel* m_copyF2Kernel;
			Buffer<int4>* m_constBuffer;
		};

		static
		Data* allocate(const Device* deviceData);

		static
		void deallocate(Data* data);

		static
		void execute( Data* data, Buffer<float4>& dst, Buffer<float4>& src, int n, Option option = PER_WI_1);

		static
		void execute( Data* data, Buffer<float2>& dst, Buffer<float2>& src, int n);

		static
		void execute( Data* data, Buffer<float>& dst, Buffer<float>& src, int n);
};


#include <AdlPrimitives/Copy/CopyHost.inl>
#include <AdlPrimitives/Copy/Copy.inl>

};

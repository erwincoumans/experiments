/*
		2011 Takahiro Harada
*/

#pragma once

#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>

namespace adl
{

class PrefixScanBase
{
	public:
		enum Option
		{
			INCLUSIVE, 
			EXCLUSIVE
		};
};


template<DeviceType TYPE>
class PrefixScan : public PrefixScanBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		enum
		{
			BLOCK_SIZE = 128
		};

		struct Data
		{
			Option m_option;
			const Device* m_device;
			Kernel* m_localScanKernel;
			Kernel* m_blockSumKernel;
			Kernel* m_propagationKernel;
			Buffer<u32>* m_workBuffer;
			Buffer<int4>* m_constBuffer[3];// todo. dx need one for each
			int m_maxSize;
		};

		static
		Data* allocate(const Device* deviceData, int maxSize, Option option = EXCLUSIVE);

		static
		void deallocate(Data* data);

		static
		void execute(Data* data, Buffer<u32>& src, Buffer<u32>& dst, int n, u32* sum = 0);
};



#include <AdlPrimitives/Scan/PrefixScanHost.inl>
#include <AdlPrimitives/Scan/PrefixScan.inl>

};

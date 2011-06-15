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


template<>
class RadixSort32<TYPE_HOST> : public RadixSort32Base
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		enum
		{
			BITS_PER_PASS = 8,
			NUM_TABLES = (1<<BITS_PER_PASS),
		};

		struct Data
		{
			HostBuffer<u32>* m_workBuffer;
		};

		static
		Data* allocate(const Device* device, int maxSize)
		{
			ADLASSERT( device->m_type == TYPE_HOST );

			Data* data = new Data;
			data->m_workBuffer = new HostBuffer<u32>( device, maxSize );
			return data;
		}

		static
		void deallocate(Data* data)
		{
			delete data->m_workBuffer;
			delete data;
		}

		static
		void execute(Data* data, Buffer<u32>& inout, int n, int sortBits = 32)
		{
			ADLASSERT( inout.getType() == TYPE_HOST );

			int tables[NUM_TABLES];
			int counter[NUM_TABLES];

			u32* src = inout.m_ptr;
			u32* dst = data->m_workBuffer->m_ptr;

			for(int startBit=0; startBit<sortBits; startBit+=BITS_PER_PASS)
			{
				for(int i=0; i<NUM_TABLES; i++)
				{
					tables[i] = 0;
				}

				for(int i=0; i<n; i++)
				{
					int tableIdx = (src[i] >> startBit) & (NUM_TABLES-1);
					tables[tableIdx]++;
				}

				//	prefix scan
				int sum = 0;
				for(int i=0; i<NUM_TABLES; i++)
				{
					int iData = tables[i];
					tables[i] = sum;
					sum += iData;
					counter[i] = 0;
				}

				//	distribute
				for(int i=0; i<n; i++)
				{
					int tableIdx = (src[i] >> startBit) & (NUM_TABLES-1);

					dst[tables[tableIdx] + counter[tableIdx]] = src[i];
					counter[tableIdx] ++;
				}

				swap2( src, dst );
			}

			{
				if( src != inout.m_ptr )
				{
					memcpy( dst, src, sizeof(u32)*n );
				}
			}

		}

		static
		void execute(Data* data, Buffer<u32>& in, Buffer<u32>& out, int n, int sortBits = 32)
		{
			ADLASSERT(0);
		}
};


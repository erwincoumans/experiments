/*
		2011 Takahiro Harada
*/

template<>
class PrefixScan<TYPE_HOST> : public PrefixScanBase
{
	public:
		struct Data
		{
			Option m_option;
		};

		static
		Data* allocate(const Device* deviceData, int maxSize, Option option = EXCLUSIVE)
		{
			ADLASSERT( deviceData->m_type == TYPE_HOST );

			Data* data = new Data;
			data->m_option = option;
			return data;
		}

		static
		void deallocate(Data* data)
		{
			delete data;
		}

		static
		void execute(Data* data, Buffer<u32>& src, Buffer<u32>& dst, int n, u32* sum = 0)
		{
			ADLASSERT( src.getType() == TYPE_HOST && dst.getType() == TYPE_HOST );
			HostBuffer<u32>& hSrc = (HostBuffer<u32>&)src;
			HostBuffer<u32>& hDst = (HostBuffer<u32>&)dst;

			u32 s = 0;
			if( data->m_option == EXCLUSIVE )
			{
				for(int i=0; i<n; i++)
				{
					hDst[i] = s;
					s += hSrc[i];
				}
			}
			else
			{
				for(int i=0; i<n; i++)
				{
					s += hSrc[i];
					hDst[i] = s;
				}
			}

			if( sum )
			{
				*sum = s;
			}
		}


};

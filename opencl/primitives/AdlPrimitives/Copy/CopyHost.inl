/*
		2011 Takahiro Harada
*/

template<>
class Copy<TYPE_HOST> : public CopyBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct Data
		{
		};

		static
		Data* allocate(const Device* deviceData)
		{
			ADLASSERT( TYPE_HOST == deviceData->m_type );
			return 0;
		}

		static
		void deallocate(Data* data)
		{
			return;
		}

		static
		void execute( Data* data, Buffer<float4>& dst, Buffer<float4>& src, int n, Option option = PER_WI_1)
		{
			ADLASSERT( TYPE_HOST == dst.getType() );
			ADLASSERT( TYPE_HOST == src.getType() );

			HostBuffer<float4>& dstH = (HostBuffer<float4>&)dst;
			HostBuffer<float4>& srcH = (HostBuffer<float4>&)src;

			for(int i=0; i<n; i++)
			{
				dstH[i] = srcH[i];
			}
		}

		static
		void execute( Data* data, Buffer<float2>& dst, Buffer<float2>& src, int n)
		{
			ADLASSERT( TYPE_HOST == dst.getType() );
			ADLASSERT( TYPE_HOST == src.getType() );

			HostBuffer<float2>& dstH = (HostBuffer<float2>&)dst;
			HostBuffer<float2>& srcH = (HostBuffer<float2>&)src;

			for(int i=0; i<n; i++)
			{
				dstH[i] = srcH[i];
			}
		}

		static
		void execute( Data* data, Buffer<float>& dst, Buffer<float>& src, int n)
		{
			ADLASSERT( TYPE_HOST == dst.getType() );
			ADLASSERT( TYPE_HOST == src.getType() );

			HostBuffer<float>& dstH = (HostBuffer<float>&)dst;
			HostBuffer<float>& srcH = (HostBuffer<float>&)src;

			for(int i=0; i<n; i++)
			{
				dstH[i] = srcH[i];
			}
		}
};


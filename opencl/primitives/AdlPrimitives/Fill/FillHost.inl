/*
		2011 Takahiro Harada
*/

template<>
class Fill<TYPE_HOST>
{
	public:
		struct Data
		{
		};

		static
		Data* allocate(const Device* deviceData)
		{
			return 0;
		}

		static
		void deallocate(Data* data)
		{

		}

		template<typename T>
		static
		void executeImpl(Data* data, Buffer<T>& src, const T& value, int n, int offset = 0)
		{
			ADLASSERT( src.getType() == TYPE_HOST );
			ADLASSERT( src.m_size >= offset+n );
			HostBuffer<T>& hSrc = (HostBuffer<T>&)src;

			for(int idx=offset; idx<offset+n; idx++)
			{
				hSrc[idx] = value;
			}
		}

		static
		void execute(Data* data, Buffer<int>& src, const int& value, int n, int offset = 0)
		{
			executeImpl( data, src, value, n, offset );
		}

		static
		void execute(Data* data, Buffer<int2>& src, const int2& value, int n, int offset = 0)
		{
			executeImpl( data, src, value, n, offset );
		}

		static
		void execute(Data* data, Buffer<int4>& src, const int4& value, int n, int offset = 0)
		{
			executeImpl( data, src, value, n, offset );
		}

/*
		static
		void execute(Data* data, Buffer<int>& src, int value, int n, int offset = 0)
		{
			ADLASSERT( src.getType() == TYPE_HOST );
			ADLASSERT( src.m_size <= offset+n );
			HostBuffer<u32>& hSrc = (HostBuffer<u32>&)src;

			for(int idx=offset; idx<offset+n; idx++)
			{
				src[i] = value;
			}
		}

		static
		void execute(Data* data, Buffer<int2>& src, const int2& value, int n, int offset = 0)
		{
			ADLASSERT( src.getType() == TYPE_HOST );
			ADLASSERT( src.m_size <= offset+n );

		}

		static
		void execute(Data* data, Buffer<int4>& src, const int4& value, int n, int offset = 0)
		{
			ADLASSERT( src.getType() == TYPE_HOST );
			ADLASSERT( src.m_size <= offset+n );

		}
*/
};


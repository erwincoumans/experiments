/*
		2011 Takahiro Harada
*/

namespace adl
{

struct DeviceHost : public Device
{
	DeviceHost() : Device( TYPE_HOST ){}

	__inline
	void initialize(const Config& cfg);
	__inline
	void release();

	template<typename T>
	__inline
	void allocate(Buffer<T>* buf, int nElems, BufferBase::BufferType type);

	template<typename T>
	__inline
	void deallocate(Buffer<T>* buf);

	template<typename T>
	__inline
	void copy(Buffer<T>* dst, const Buffer<T>* src, int nElems);

	template<typename T>
	__inline
	void copy(T* dst, const Buffer<T>* src, int nElems, int offsetNElems = 0);

	template<typename T>
	__inline
	void copy(Buffer<T>* dst, const T* src, int nElems, int offsetNElems = 0);

	__inline
	void waitForCompletion() const;
};

void DeviceHost::initialize(const Config& cfg)
{

}

void DeviceHost::release()
{

}

template<typename T>
void DeviceHost::allocate(Buffer<T>* buf, int nElems, BufferBase::BufferType type)
{
	buf->m_device = this;

	if( type == BufferBase::BUFFER_CONST ) return;

	buf->m_ptr = new T[nElems];
	ADLASSERT( buf->m_ptr );
	buf->m_size = nElems;
}

template<typename T>
void DeviceHost::deallocate(Buffer<T>* buf)
{
	if( buf->m_ptr ) delete [] buf->m_ptr;
}

template<typename T>
void DeviceHost::copy(Buffer<T>* dst, const Buffer<T>* src, int nElems)
{
	copy( dst, src->m_ptr, nElems );
}

template<typename T>
void DeviceHost::copy(T* dst, const Buffer<T>* src, int nElems, int srcOffsetNElems)
{
	ADLASSERT( src->getType() == TYPE_HOST );
	memcpy( dst, src->m_ptr+srcOffsetNElems, nElems*sizeof(T) );
}

template<typename T>
void DeviceHost::copy(Buffer<T>* dst, const T* src, int nElems, int dstOffsetNElems)
{
	ADLASSERT( dst->getType() == TYPE_HOST );
	memcpy( dst->m_ptr+dstOffsetNElems, src, nElems*sizeof(T) );
}

void DeviceHost::waitForCompletion() const
{

}

};

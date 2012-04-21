#ifndef BT_OPENCL_ARRAY_H
#define BT_OPENCL_ARRAY_H

#include "LinearMath/btAlignedObjectArray.h"
#include "../basic_initialize/btOpenCLInclude.h"

template <typename T> 
class btOpenCLArray
{
	int	m_size;
	int	m_capacity;
	cl_mem	m_clBuffer;

	cl_context		 m_clContext;
	cl_command_queue m_commandQueue;

	void deallocate()
	{
		if (m_clBuffer)
		{
			clReleaseMemObject(m_clBuffer);
			m_clBuffer = 0;
		}
	}

	btOpenCLArray<T>& operator=(const btOpenCLArray<T>& src);

public:

	btOpenCLArray(cl_context ctx, cl_command_queue queue)
	:m_size(0), m_capacity(0),m_clBuffer(0),m_clContext(ctx),m_commandQueue(queue)
	{
	}

	
	
// we could enable this assignment, but need to make sure to avoid accidental deep copies
//	btOpenCLArray<T>& operator=(const btAlignedObjectArray<T>& src) 
//	{
//		copyFromArray(src);
//		return *this;
//	}


	cl_mem	getBufferCL()
	{
		return m_clBuffer;
	}

	
	virtual ~btOpenCLArray()
	{
		deallocate();
		m_size=0;
		m_capacity=0;
	}
	


	SIMD_FORCE_INLINE	void	resize(int newsize, bool copyOldContents=true)
	{
		int curSize = size();

		if (newsize < curSize)
		{
			//leave the OpenCL memory for now
		} else
		{
			if (newsize > size())
			{
				reserve(newsize,copyOldContents);
			}

			//leave new data uninitialized (init in debug mode?)
			//for (int i=curSize;i<newsize;i++) ...
		}

		m_size = newsize;
	}

	SIMD_FORCE_INLINE int size() const
	{
		return m_size;
	}

	SIMD_FORCE_INLINE	int capacity() const
	{	
		return m_capacity;
	}

	SIMD_FORCE_INLINE	void reserve(int _Count, bool copyOldContents=true)
	{	// determine new minimum length of allocated storage
		if (capacity() < _Count)
		{	// not enough room, reallocate

			cl_int ciErrNum;
			//create a new OpenCL buffer
			int memSizeInBytes = sizeof(T)*_Count;
			cl_mem buf = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, memSizeInBytes, NULL, &ciErrNum);
			btAssert(ciErrNum==CL_SUCCESS);

			if (copyOldContents)
				copy(0, size(), buf);

			//deallocate the old buffer
			deallocate();

			m_clBuffer = buf;
			
			m_capacity = _Count;
		}
	}


	void copy(int beginElement, int endElement, cl_mem destination, int dstOffsetBytes=0) const
	{
		int numElements = endElement-beginElement;
		if (numElements<=0)
			return;

		btAssert(m_clBuffer);
		btAssert(destination);
		
		//likely some error, destination is same as source
		btAssert(m_clBuffer != destination);

		btAssert(endElement<=m_size);
		
		cl_int status = 0;
		

		btAssert(numElements>0);
		btAssert(numElements<=m_size);

		int srcOffsetBytes = sizeof(T)*beginElement;
		
		status = clEnqueueCopyBuffer( m_commandQueue, m_clBuffer, destination, 
			srcOffsetBytes, dstOffsetBytes, sizeof(T)*numElements, 0, 0, 0 );

		btAssert( status == CL_SUCCESS );
	}

	void copyFromArray(const btAlignedObjectArray<T>& srcArray, bool waitForCompletion=true)
	{
		int newSize = srcArray.size();
		
		bool copyOldContents = false;
		resize (newSize,copyOldContents);

		copyFromHost(0,newSize,&srcArray[0]);

		if (waitForCompletion)
			clFinish(m_commandQueue);
	}

	void copyFromHost(int firstElem, int lastElem, const T* src)
	{
		cl_int status = 0;
		int numElems = lastElem - firstElem;

		int sizeInBytes=sizeof(T)*numElems;
		status = clEnqueueWriteBuffer( m_commandQueue, m_clBuffer, 0, sizeof(T)*firstElem, sizeInBytes,
		src, 0,0,0 );
		btAssert(status == CL_SUCCESS );
	}
	

	void copyToArray(btAlignedObjectArray<T>& destArray, bool waitForCompletion=true) const
	{
		destArray.resize(this->size());

		copyToHost(0,size(),&destArray[0]);
		
		if (waitForCompletion)
			clFinish(m_commandQueue);
	}

	void copyToHost(int firstElem, int lastElem, T* destPtr) const
	{
		int nElems = lastElem-firstElem;
		cl_int status = 0;
		status = clEnqueueReadBuffer( m_commandQueue, m_clBuffer, 0, sizeof(T)*firstElem, sizeof(T)*nElems,
		destPtr, 0,0,0 );
		btAssert( status==CL_SUCCESS );
	}
	
	void copyFromOpenCLArray(const btOpenCLArray& src)
	{
		int newSize = src.size();
		resize(newSize);
		if (size())
		{
			src.copy(0,size(),this->m_clBuffer);
		}

	}

};


#endif //BT_OPENCL_ARRAY_H

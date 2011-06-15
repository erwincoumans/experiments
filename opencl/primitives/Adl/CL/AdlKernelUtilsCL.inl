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

struct KernelCL : public Kernel
{
	cl_kernel& getKernel() { return (cl_kernel&)m_kernel; }
};



template<>
void KernelBuilder<TYPE_CL>::setFromFile( const Device* deviceData, const char* fileName, const char* option, bool addExtension )
{
	m_deviceData = deviceData;

	char fileNameWithExtension[256];

	if( addExtension )
		sprintf_s( fileNameWithExtension, "%s.cl", fileName );
	else
		sprintf_s( fileNameWithExtension, "%s", fileName );

	class File
	{
		public:
			__inline
			bool open(const char* fileNameWithExtension)
			{
				size_t      size;
				char*       str;

				// Open file stream
				std::fstream f(fileNameWithExtension, (std::fstream::in | std::fstream::binary));

				// Check if we have opened file stream
				if (f.is_open()) {
					size_t  sizeFile;
					// Find the stream size
					f.seekg(0, std::fstream::end);
					size = sizeFile = (size_t)f.tellg();
					f.seekg(0, std::fstream::beg);

					str = new char[size + 1];
					if (!str) {
						f.close();
						return  NULL;
					}

					// Read file
					f.read(str, sizeFile);
					f.close();
					str[size] = '\0';

					m_source  = str;

					delete[] str;

					return true;
				}

				return false;
			}
			const std::string& getSource() const {return m_source;}

		private:
			std::string m_source;
	};

	cl_program& program = (cl_program&)m_ptr;
	cl_int status = 0;
	File kernelFile;
	ADLASSERT( kernelFile.open( fileNameWithExtension ) );
	const char* source = kernelFile.getSource().c_str();

	setFromSrc( m_deviceData, source, option );
}

template<>
void KernelBuilder<TYPE_CL>::setFromSrc( const Device* deviceData, const char* src, const char* option )
{
	ADLASSERT( deviceData->m_type == TYPE_CL );
	m_deviceData = deviceData;
	const DeviceCL* dd = (const DeviceCL*) deviceData;

	cl_program& program = (cl_program&)m_ptr;
	cl_int status = 0;
	size_t srcSize[] = {strlen( src )};
	program = clCreateProgramWithSource( dd->m_context, 1, &src, srcSize, &status );
	ADLASSERT( status == CL_SUCCESS );
	status = clBuildProgram( program, 1, &dd->m_deviceIdx, option, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		char *build_log;
		size_t ret_val_size;
		clGetProgramBuildInfo(program, dd->m_deviceIdx, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
		build_log = new char[ret_val_size+1];
		clGetProgramBuildInfo(program, dd->m_deviceIdx, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

		build_log[ret_val_size] = '\0';

		adlDebugPrintf("%s\n", build_log);

		delete build_log;
		ADLASSERT(0);
	}
}

template<>
KernelBuilder<TYPE_CL>::~KernelBuilder()
{
	cl_program program = (cl_program)m_ptr;
	clReleaseProgram( program );
}

template<>
void KernelBuilder<TYPE_CL>::createKernel( const char* funcName, Kernel& kernelOut )
{
	KernelCL* clKernel = (KernelCL*)&kernelOut;

	cl_program program = (cl_program)m_ptr;
	cl_int status = 0;
	clKernel->getKernel() = clCreateKernel(program, funcName, &status );
	ADLASSERT( status == CL_SUCCESS );

	kernelOut.m_type = TYPE_CL;
}

template<>
void KernelBuilder<TYPE_CL>::deleteKernel( Kernel& kernel )
{
	KernelCL* clKernel = (KernelCL*)&kernel;
	clReleaseKernel( clKernel->getKernel() );
}



class LauncherCL
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		__inline
		static void setBuffers( Launcher* launcher, BufferInfo* buffInfo, int n );
		template<typename T>
		__inline
		static void setConst( Launcher* launcher, Buffer<T>& constBuff, const T& consts );
		__inline
		static void launch2D( Launcher* launcher, int numThreadsX, int numThreadsY, int localSizeX, int localSizeY );
};

void LauncherCL::setBuffers( Launcher* launcher, BufferInfo* buffInfo, int n )
{
	KernelCL* clKernel = (KernelCL*)launcher->m_kernel;
	for(int i=0; i<n; i++)
	{
		Buffer<int>* buff = (Buffer<int>*)buffInfo[i].m_buffer;
		cl_int status = clSetKernelArg( clKernel->getKernel(), launcher->m_idx++, sizeof(cl_mem), &buff->m_ptr );
		ADLASSERT( status == CL_SUCCESS );
	}
}

template<typename T>
void LauncherCL::setConst( Launcher* launcher, Buffer<T>& constBuff, const T& consts )
{
	KernelCL* clKernel = (KernelCL*)launcher->m_kernel;
	cl_int status = clSetKernelArg( clKernel->getKernel(), launcher->m_idx++, sizeof(T), &consts );
	ADLASSERT( status == CL_SUCCESS );
}

void LauncherCL::launch2D( Launcher* launcher, int numThreadsX, int numThreadsY, int localSizeX, int localSizeY )
{
	KernelCL* clKernel = (KernelCL*)launcher->m_kernel;
	const DeviceCL* ddcl = (const DeviceCL*)launcher->m_deviceData;
	size_t gRange[3] = {1,1,1};
	size_t lRange[3] = {1,1,1};
	lRange[0] = localSizeX;
	lRange[1] = localSizeY;
	gRange[0] = max((size_t)1, (numThreadsX/lRange[0])+(!(numThreadsX%lRange[0])?0:1));
	gRange[0] *= lRange[0];
	gRange[1] = max((size_t)1, (numThreadsY/lRange[1])+(!(numThreadsY%lRange[1])?0:1));
	gRange[1] *= lRange[1];

	cl_int status = clEnqueueNDRangeKernel( ddcl->m_commandQueue, 
		clKernel->getKernel(), 2, NULL, gRange, lRange, 0,0,0 );
	ADLASSERT( status == CL_SUCCESS );
}




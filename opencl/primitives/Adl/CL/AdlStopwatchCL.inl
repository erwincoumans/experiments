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

struct StopwatchCL : public StopwatchBase
{
	public:
		__inline
		StopwatchCL() : StopwatchBase(){}
		__inline
		~StopwatchCL();

		__inline
		void init( const Device* deviceData );
		__inline
		void start();
		__inline
		void split();
		__inline
		void stop();
		__inline
		float getMs();
		__inline
		void getMs( float* times, int capacity );

	public:
		LARGE_INTEGER m_t[CAPACITY];
};

StopwatchCL::~StopwatchCL()
{

}

void StopwatchCL::init( const Device* deviceData )
{
	m_device = deviceData;
}

void StopwatchCL::start()
{
	m_idx = 0;
	split();
}

void StopwatchCL::split()
{
	DeviceUtils::waitForCompletion( m_device );
	QueryPerformanceCounter(&m_t[m_idx++]);
}

void StopwatchCL::stop()
{
	split();
}

float StopwatchCL::getMs()
{
	LARGE_INTEGER m_frequency;
	QueryPerformanceFrequency( &m_frequency );
	return (float)(1000*(m_t[1].QuadPart - m_t[0].QuadPart))/m_frequency.QuadPart;
}

void StopwatchCL::getMs( float* times, int capacity )
{
	LARGE_INTEGER m_frequency;
	QueryPerformanceFrequency( &m_frequency );

	for(int i=0; i<capacity; i++) times[i] = 0.f;

	for(int i=0; i<min(capacity, m_idx); i++)
	{
		times[i] = (float)(1000*(m_t[i+1].QuadPart - m_t[i].QuadPart))/m_frequency.QuadPart;
	}
}

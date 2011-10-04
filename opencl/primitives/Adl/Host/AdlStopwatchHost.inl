/*
		2011 Takahiro Harada
*/

namespace adl
{

class StopwatchHost : public StopwatchBase
{
	public:
		__inline
		StopwatchHost();
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

	private:
		LARGE_INTEGER m_frequency;
		LARGE_INTEGER m_t[CAPACITY];
};

__inline
StopwatchHost::StopwatchHost()
 : StopwatchBase()
{
}

__inline
void StopwatchHost::init( const Device* deviceData )
{
	m_device = deviceData;
	QueryPerformanceFrequency( &m_frequency );
}

__inline
void StopwatchHost::start()
{
	m_idx = 0;
	QueryPerformanceCounter(&m_t[m_idx++]);
}

__inline
void StopwatchHost::split()
{
	QueryPerformanceCounter(&m_t[m_idx++]);
}

__inline
void StopwatchHost::stop()
{
	split();
}

__inline
float StopwatchHost::getMs()
{
	return (float)(1000*(m_t[1].QuadPart - m_t[0].QuadPart))/m_frequency.QuadPart;
}

__inline
void StopwatchHost::getMs(float* times, int capacity)
{
	for(int i=0; i<capacity; i++) times[i] = 0.f;

	for(int i=0; i<min(capacity, m_idx-1); i++)
	{
		times[i] = (float)(1000*(m_t[i+1].QuadPart - m_t[i].QuadPart))/m_frequency.QuadPart;
	}
}

};
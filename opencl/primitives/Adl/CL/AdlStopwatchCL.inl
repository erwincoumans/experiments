/*
		2011 Takahiro Harada
*/

namespace adl
{

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

};
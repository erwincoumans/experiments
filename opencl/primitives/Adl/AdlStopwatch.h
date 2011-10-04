/*
		2011 Takahiro Harada
*/

#include <windows.h>

namespace adl
{

struct StopwatchBase
{
	__inline
	StopwatchBase(): m_device(0){}
	__inline
	StopwatchBase( const Device* deviceData ){ init(deviceData); }
	__inline
	virtual ~StopwatchBase(){}

	__inline
	virtual void init( const Device* deviceData ) = 0;
	__inline
	virtual void start() = 0;
	__inline
	virtual void split() = 0;
	__inline
	virtual void stop() = 0;
	__inline
	virtual float getMs() = 0;
	__inline
	virtual void getMs( float* times, int capacity ) = 0;
	__inline
	int getNIntervals() const{ return m_idx-1;}

	enum
	{
		CAPACITY = 64,
	};

	const Device* m_device;
	int m_idx;
};

struct Stopwatch
{
	__inline
	Stopwatch( const Device* deviceData = NULL ) { m_impl=0; if(deviceData) init(deviceData);}
	__inline
	~Stopwatch();

	__inline
	void init( const Device* deviceData );
	__inline
	void start(){if(!m_impl) init(0); m_impl->start();}
	__inline
	void split(){m_impl->split();}
	__inline
	void stop(){m_impl->stop();}
	__inline
	float getMs(){ return m_impl->getMs();}
	__inline
	void getMs( float* times, int capacity ){m_impl->getMs(times, capacity);}
	__inline
	int getNIntervals() const{return m_impl->getNIntervals();}

	StopwatchBase* m_impl;
};

};

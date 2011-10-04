/*
		2011 Takahiro Harada
*/

namespace adl
{

void Stopwatch::init( const Device* deviceData )
{
	ADLASSERT( m_impl == 0 );

	if( deviceData )
	{
		switch( deviceData->m_type )
		{
#if defined(ADL_ENABLE_CL)
		case TYPE_CL:
			m_impl = new StopwatchCL;
			break;
#endif
#if defined(ADL_ENABLE_DX11)
		case TYPE_DX11:
			m_impl = new StopwatchDX11;
			break;
#endif
		case TYPE_HOST:
			m_impl = new StopwatchHost;
			break;
		default:
			ADLASSERT(0);
			break;
		};
	}
	else
	{
		m_impl = new StopwatchHost;
	}
	m_impl->init( deviceData );
}

Stopwatch::~Stopwatch()
{
	if( m_impl == 0 ) return;
	delete m_impl;
}

};
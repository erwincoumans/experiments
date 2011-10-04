/*
		2011 Takahiro Harada
*/

namespace adl
{

struct StopwatchDX11 : public StopwatchBase
{
	public:
		__inline
		StopwatchDX11() : StopwatchBase(){}
		__inline
		~StopwatchDX11();

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
		ID3D11Query* m_tQuery[CAPACITY+1];
		ID3D11Query* m_fQuery;
		UINT64 m_t[CAPACITY];
};

void StopwatchDX11::init( const Device* deviceData )
{
	ADLASSERT( deviceData->m_type == TYPE_DX11 );
	m_device = deviceData;
	{
		D3D11_QUERY_DESC qDesc;
		qDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
		qDesc.MiscFlags = 0;
		((const DeviceDX11*)m_device)->m_device->CreateQuery( &qDesc, &m_fQuery );
	}
	for(int i=0; i<CAPACITY+1; i++)
	{
		D3D11_QUERY_DESC qDesc;
		qDesc.Query = D3D11_QUERY_TIMESTAMP;
		qDesc.MiscFlags = 0;
		((const DeviceDX11*)m_device)->m_device->CreateQuery( &qDesc, &m_tQuery[i] );
	}
}

StopwatchDX11::~StopwatchDX11()
{
	m_fQuery->Release();
	for(int i=0; i<CAPACITY+1; i++)
	{
		m_tQuery[i]->Release();
	}
}

void StopwatchDX11::start()
{
	m_idx = 0;
	((const DeviceDX11*)m_device)->m_context->Begin( m_fQuery );
	((const DeviceDX11*)m_device)->m_context->End( m_tQuery[m_idx++] );
}

void StopwatchDX11::split()
{
	if( m_idx < CAPACITY )
		((const DeviceDX11*)m_device)->m_context->End( m_tQuery[m_idx++] );
}

void StopwatchDX11::stop()
{
	((const DeviceDX11*)m_device)->m_context->End( m_tQuery[m_idx++] );
	((const DeviceDX11*)m_device)->m_context->End( m_fQuery );
}

float StopwatchDX11::getMs()
{
	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
//	m_deviceData->m_context->End( m_fQuery );
	while( ((const DeviceDX11*)m_device)->m_context->GetData( m_fQuery, &d,sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT),0 ) == S_FALSE ) {}

	while( ((const DeviceDX11*)m_device)->m_context->GetData( m_tQuery[0], &m_t[0],sizeof(UINT64),0 ) == S_FALSE ){}
	while( ((const DeviceDX11*)m_device)->m_context->GetData( m_tQuery[1], &m_t[1],sizeof(UINT64),0 ) == S_FALSE ){}

	ADLASSERT( d.Disjoint == false );

	float elapsedMs = (m_t[1] - m_t[0])/(float)d.Frequency*1000;
	return elapsedMs;

}

void StopwatchDX11::getMs( float* times, int capacity )
{
	ADLASSERT( capacity <= CAPACITY );

	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
	while( ((const DeviceDX11*)m_device)->m_context->GetData( m_fQuery, &d,sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT),0 ) == S_FALSE ) {}

	for(int i=0; i<m_idx; i++)
	{
		while( ((const DeviceDX11*)m_device)->m_context->GetData( m_tQuery[i], &m_t[i],sizeof(UINT64),0 ) == S_FALSE ){}
	}

	ADLASSERT( d.Disjoint == false );

	for(int i=0; i<capacity; i++)
	{
		times[i] = (m_t[i+1] - m_t[i])/(float)d.Frequency*1000;
	}
}

};
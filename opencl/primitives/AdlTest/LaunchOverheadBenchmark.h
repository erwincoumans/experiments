#include <AdlPrimitives/Copy/Copy.h>



template<DeviceType TYPE>
__inline
void copyTest( Device* device )
{
	int size = 65*1024;

	Buffer<float4> buf0( device, size );
	Buffer<float4> buf1( device, size );

	Stopwatch sw( device );
	
	Copy<TYPE>::Data* data = Copy<TYPE>::allocate( device );

	for(int i=0; i<10; i++)
		Copy<TYPE>::execute( data, buf1, buf0, size, CopyBase::PER_WI_1 );
	DeviceUtils::waitForCompletion( device );

	{
		const int nTests = 12;

		float t[nTests];

		for(int ii=0; ii<nTests; ii++)
		{
			int iter = 1<<ii;

			DeviceUtils::waitForCompletion( device );
			sw.start();
			for(int i=0; i<iter; i++)
			{
				Copy<TYPE>::execute( data, buf1, buf0, size, CopyBase::PER_WI_1 );
			}
			DeviceUtils::waitForCompletion( device );
			sw.stop();

			t[ii] = sw.getMs()/(float)iter;
		}

		for(int ii=0; ii<nTests; ii++)
		{
			printf("%d:	%3.4fms	(%3.2fGB/s)\n", (1<<ii), t[ii], size*16*2/1024.f/1024.f/t[ii]);
		}
		printf("\n");

	}
	
	Copy<TYPE>::deallocate( data );
}

void launchOverheadBenchmark()
{
	printf("LaunchOverheadBenchmark\n");

	AdlAllocate();

	Device* ddcl;
#if defined(ADL_ENABLE_DX11)
	Device* dddx;
#endif
	{
		DeviceUtils::Config cfg;
		ddcl = DeviceUtils::allocate( TYPE_CL, cfg );
#if defined(ADL_ENABLE_DX11)
		dddx = DeviceUtils::allocate( TYPE_DX11, cfg );
#endif
	}

	{
		printf("CL\n");
		copyTest<TYPE_CL>( ddcl );
	}
#if defined(ADL_ENABLE_DX11)
	{
		printf("DX11\n");
		copyTest<TYPE_DX11>( dddx );
	}
#endif

	AdlDeallocate();
}


//1, 2, 4, 8, 16, 32, 64, 128, 256, 


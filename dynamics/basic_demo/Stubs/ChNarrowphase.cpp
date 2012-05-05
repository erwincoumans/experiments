/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Takahiro Harada


//#define PATH "..\\..\\dynamics\\basic_demo\\Stubs\\ChNarrowphaseKernels"
#define NARROWPHASE_KERNEL_PATH "..\\../dynamics/basic_demo/Stubs/ChNarrowphaseKernels.cl"
#define KERNEL0 "SupportCullingKernel"
#define KERNEL1 "NarrowphaseKernel"

#include "../../../opencl/broadphase_benchmark/btLauncherCL.h"
#include "../../../opencl/basic_initialize/btOpenCLUtils.h"

#include "ChNarrowphase.h"

#include "ChNarrowphaseKernels.h"

class ChNarrowphaseImp
{
public:
	static
	__inline
	u32 u32Pack(u8 x, u8 y, u8 z, u8 w)
	{
		return (x) | (y<<8) | (z<<16) | (w<<24);
	}

};


ChNarrowphase::ChNarrowphase(cl_context ctx, cl_device_id device, cl_command_queue queue)
	:m_context(ctx),
	m_device(device),
	m_queue(queue)
{

//	sprintf(options, "-I .\\NarrowPhaseCL\\");
	
	const char* additionalMacros = "";
	const char* srcFileNameForCaching="";

	cl_int pErrNum;
	char* kernelSource = 0;
	
	cl_program narrowphaseProg = btOpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, NARROWPHASE_KERNEL_PATH);
	btAssert(narrowphaseProg);

	m_supportCullingKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "SupportCullingKernel", &pErrNum, narrowphaseProg,additionalMacros );
	btAssert(m_supportCullingKernel );

	m_narrowphaseKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "NarrowphaseKernel", &pErrNum, narrowphaseProg,additionalMacros );
	btAssert(m_narrowphaseKernel );

	m_narrowphaseWithPlaneKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "NarrowphaseWithPlaneKernel", &pErrNum, narrowphaseProg,additionalMacros );
	btAssert(m_narrowphaseWithPlaneKernel );

	m_counterBuffer = new btOpenCLArray<u32>(ctx,queue, 1 );

}

ChNarrowphase::~ChNarrowphase()
{
	delete m_counterBuffer;
	clReleaseKernel(m_supportCullingKernel);
	clReleaseKernel(m_narrowphaseKernel);
	clReleaseKernel(m_narrowphaseWithPlaneKernel);

}



#if 0
void ChNarrowphase::setShape( ShapeDataType shapeBuf, ShapeBase* shape, int idx, float collisionMargin )
{
	ConvexHeightField* cvxShape = new ConvexHeightField( shape );
	btOpenCLArray<ShapeData>* dst = (btOpenCLArray<ShapeData>*)shapeBuf;
	cvxShape->m_aabb.expandBy( make_float4( collisionMargin ) );
	{
		ShapeData s;
		{
			for(int j=0; j<HEIGHT_RES*HEIGHT_RES*6; j++)
			{
				s.m_normal[j] = cvxShape->m_normal[j];
			}
			for(int j=0; j<HEIGHT_RES*HEIGHT_RES*6/4; j++)
			{
				s.m_height4[j] = ChNarrowphaseImp::u32Pack( cvxShape->m_data[4*j], cvxShape->m_data[4*j+1], cvxShape->m_data[4*j+2], cvxShape->m_data[4*j+3] );
				s.m_supportHeight4[j] = ChNarrowphaseImp::u32Pack( cvxShape->m_supportHeight[4*j], cvxShape->m_supportHeight[4*j+1], cvxShape->m_supportHeight[4*j+2], cvxShape->m_supportHeight[4*j+3] );
			}
			s.m_scale = cvxShape->m_scale;
		}
		dst->copyFromHostPointer( &s, 1, idx );
		clFinish(m_queue);
	}
	delete cvxShape;
}
#endif

void ChNarrowphase::setShape( btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf, ConvexHeightField* cvxShape, int idx, float collisionMargin )
{
	btOpenCLArray<ShapeData>* dst = (btOpenCLArray<ShapeData>*)shapeBuf;
	cvxShape->m_aabb.expandBy( make_float4( collisionMargin,collisionMargin,collisionMargin,0.f ) );
	{
		ShapeData s;
		{
			for(int j=0; j<HEIGHT_RES*HEIGHT_RES*6; j++)
			{
				s.m_normal[j] = cvxShape->m_normal[j];
			}
			for(int j=0; j<HEIGHT_RES*HEIGHT_RES*6/4; j++)
			{
				s.m_height4[j] = ChNarrowphaseImp::u32Pack( cvxShape->m_data[4*j], cvxShape->m_data[4*j+1], cvxShape->m_data[4*j+2], cvxShape->m_data[4*j+3] );
				s.m_supportHeight4[j] = ChNarrowphaseImp::u32Pack( cvxShape->m_supportHeight[4*j], cvxShape->m_supportHeight[4*j+1], cvxShape->m_supportHeight[4*j+2], cvxShape->m_supportHeight[4*j+3] );
			}
			s.m_scale = cvxShape->m_scale;
		}
		dst->copyFromHostPointer( &s, 1, idx );
		clFinish(m_queue);
	}
}

// Run NarrowphaseKernel
void ChNarrowphase::execute( const btOpenCLArray<int2>* pairs, int nPairs, const btOpenCLArray<RigidBodyBase::Body>* bodyBuf,
			const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const Config& cfg )
{
	if( nPairs == 0 ) return;


	btOpenCLArray<ShapeData>* shapeBuffer = (btOpenCLArray<ShapeData>*)shapeBuf;
	
	ConstData cdata;
	cdata.m_nPairs = nPairs;
	cdata.m_collisionMargin = cfg.m_collisionMargin;
	cdata.m_capacity = contactOut->capacity()  - nContacts;

	u32 n = nContacts;
	m_counterBuffer->copyFromHostPointer( &n, 1 );
//	DeviceUtils::waitForCompletion( device );

	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( pairs->getBufferCL(), true ), btBufferInfoCL( shapeBuf->getBufferCL()), btBufferInfoCL( bodyBuf->getBufferCL()), 
			btBufferInfoCL( contactOut->getBufferCL()),
			btBufferInfoCL( m_counterBuffer->getBufferCL() ) };
		btLauncherCL launcher(m_queue,m_narrowphaseKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst(  cdata );
		launcher.launch1D( nPairs*64, 64 );
	}

	m_counterBuffer->copyToHostPointer( &n, 1 );
	clFinish(m_queue);

	nContacts = btMin((int)n, contactOut->capacity() );

}

// Run NarrowphaseWithPlaneKernel
void ChNarrowphase::execute( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			const btOpenCLArray<float4>* vtxBuf, const btOpenCLArray<int4>* idxBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const Config& cfg )
{
	if( nPairs == 0 ) return;

	btOpenCLArray<ShapeData>* shapeBuffer = (btOpenCLArray<ShapeData>*)shapeBuf;

	

	ConstData cdata;
	cdata.m_nPairs = nPairs;
	cdata.m_collisionMargin = cfg.m_collisionMargin;
	cdata.m_capacity = contactOut->capacity() - nContacts;

	u32 n = nContacts;
	m_counterBuffer->copyFromHostPointer( &n, 1 );
//	DeviceUtils::waitForCompletion( device );

	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( pairs->getBufferCL(), true ), btBufferInfoCL( shapeBuf->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL()), 
			btBufferInfoCL( contactOut->getBufferCL() ),
			btBufferInfoCL( m_counterBuffer->getBufferCL() ) };
		btLauncherCL launcher(m_queue, m_narrowphaseWithPlaneKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst( cdata );
		launcher.launch1D( nPairs*64, 64 );
	}

	m_counterBuffer->copyToHostPointer( &n, 1 );
	clFinish(m_queue);


	nContacts = min2((int)n, contactOut->capacity() );


}

// Run SupportCullingKernel
int ChNarrowphase::culling(  const btOpenCLArray<int2>* pairs, int nPairs, const btOpenCLArray<RigidBodyBase::Body>* bodyBuf,
			const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf, const btOpenCLArray<int2>* pairsOut, const Config& cfg )
{
	if( nPairs == 0 ) return 0;


	btOpenCLArray<ShapeData>* shapeBuffer = (btOpenCLArray<ShapeData>*)shapeBuf;
	

	//
	btOpenCLArray<ConstData> constBuffer( this->m_context,m_queue,1);

	ConstData cdata;
	cdata.m_nPairs = nPairs;
	cdata.m_collisionMargin = cfg.m_collisionMargin;
	cdata.m_capacity = pairsOut->capacity();//or size?>getSize();

	u32 n = 0;
	m_counterBuffer->copyFromHostPointer( &n, 1 );
//	DeviceUtils::waitForCompletion( device );
	{
		btBufferInfoCL bInfo[] = { btBufferInfoCL( pairs->getBufferCL(), true ), btBufferInfoCL( shapeBuf->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL() ), 
			btBufferInfoCL( pairsOut->getBufferCL() ), btBufferInfoCL( m_counterBuffer->getBufferCL() ) };
		btLauncherCL launcher(m_queue, m_supportCullingKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst( cdata );
		launcher.launch1D( nPairs, 64 );
	}
	m_counterBuffer->copyToHostPointer( &n, 1 );
	clFinish(m_queue);
/*
	if( gPairsInNative != pairs ) delete gPairsInNative;
	if( gBodyInNative != bodyBuf ) delete gBodyInNative;
	if( gPairsOutNative != pairsOut ) 
	{
		gPairsOutNative->read( pairsOut->m_ptr, n );
		DeviceUtils::waitForCompletion( device );
		delete gPairsOutNative;
	}
*/

	return btMin((int)n, pairsOut->capacity() );//capacity or size?

}

#undef PATH
#undef KERNEL0
#undef KERNEL1
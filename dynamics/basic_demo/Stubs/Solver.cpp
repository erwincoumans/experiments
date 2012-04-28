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


#include "Solver.h"

#define SOLVER_KERNEL_PATH "../../dynamics/basic_demo/Stubs/SolverKernels.cl"
#define BATCHING_PATH "../../dynamics/basic_demo/Stubs/batchingKernels.cl"


#include "SolverKernels.h"
#include "batchingKernels.h"
#include "LinearMath/btQuickprof.h"
#include "../../opencl/broadphase_benchmark/btLauncherCL.h"

struct SolverDebugInfo
{
	int m_valInt0;
	int m_valInt1;
	int m_valInt2;
	int m_valInt3;
	
	int m_valInt4;
	int m_valInt5;
	int m_valInt6;
	int m_valInt7;

	int m_valInt8;
	int m_valInt9;
	int m_valInt10;
	int m_valInt11;

	int	m_valInt12;
	int	m_valInt13;
	int	m_valInt14;
	int	m_valInt15;


	float m_val0;
	float m_val1;
	float m_val2;
	float m_val3;
};




class SolverDeviceInl
{
public:
	struct ParallelSolveData
	{
		btOpenCLArray<u32>* m_numConstraints;
		btOpenCLArray<u32>* m_offsets;
	};
};



Solver::Solver(cl_context ctx, cl_device_id device, cl_command_queue queue, int pairCapacity)
			:m_nIterations(4),
			m_context(ctx),
			m_device(device),
			m_queue(queue)
{
	m_sort32 = new btRadixSort32CL(ctx,device,queue);
	m_scan = new btPrefixScanCL(ctx,device,queue,N_SPLIT*N_SPLIT);
	m_search = new btBoundSearchCL(ctx,device,queue,N_SPLIT*N_SPLIT);

	const int sortSize = NEXTMULTIPLEOF( pairCapacity, 512 );

	m_sortDataBuffer = new btOpenCLArray<btSortData>(ctx,queue,sortSize);
	m_contactBuffer = new btOpenCLArray<Contact4>(ctx,queue);

	m_numConstraints = new btOpenCLArray<u32>(ctx,queue,N_SPLIT*N_SPLIT );
	m_offsets = new btOpenCLArray<u32>( ctx,queue, N_SPLIT*N_SPLIT );

	const char* additionalMacros = "";
	const char* srcFileNameForCaching="";

	cl_int pErrNum;
	char* kernelSource = 0;
	
	{
		cl_program batchingProg = btOpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, BATCHING_PATH);
		btAssert(batchingProg);

		m_batchingKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "CreateBatches", &pErrNum, batchingProg,additionalMacros );
		btAssert(m_batchingKernel);
	}
	
	{
		cl_program solverProg= btOpenCLUtils::compileCLProgramFromString( ctx, device, kernelSource, &pErrNum,additionalMacros, SOLVER_KERNEL_PATH);
		btAssert(solverProg);
		
		m_batchSolveKernel= btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "BatchSolveKernel", &pErrNum, solverProg,additionalMacros );
		btAssert(m_batchSolveKernel);
	
		
		m_contactToConstraintKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "ContactToConstraintKernel", &pErrNum, solverProg,additionalMacros );
		btAssert(m_contactToConstraintKernel);
			
		m_setSortDataKernel =  btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "SetSortDataKernel", &pErrNum, solverProg,additionalMacros );
		btAssert(m_setSortDataKernel);
				
		m_reorderContactKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "ReorderContactKernel", &pErrNum, solverProg,additionalMacros );
		btAssert(m_reorderContactKernel);
		

		m_copyConstraintKernel = btOpenCLUtils::compileCLKernelFromString( ctx, device, kernelSource, "CopyConstraintKernel", &pErrNum, solverProg,additionalMacros );
		btAssert(m_copyConstraintKernel);
		
	}

			
}
		
Solver::~Solver()
{
	delete m_sortDataBuffer;
	delete m_contactBuffer;

	delete m_sort32;
	delete m_scan;
	delete m_search;


	clReleaseKernel(m_batchingKernel);
	clReleaseKernel( m_batchSolveKernel);
	clReleaseKernel( m_contactToConstraintKernel);
	clReleaseKernel( m_setSortDataKernel);
	clReleaseKernel( m_reorderContactKernel);
	clReleaseKernel( m_copyConstraintKernel);
			
}


 


void Solver::reorderConvertToConstraints( const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, 
	const btOpenCLArray<RigidBodyBase::Inertia>* shapeBuf,
	btOpenCLArray<Contact4>* contactsIn, SolverData contactCOut, void* additionalData, 
	int nContacts, const Solver::ConstraintCfg& cfg )
{
	if( m_contactBuffer )
	{
		m_contactBuffer->resize(nContacts);
	}
	if( m_contactBuffer == 0 )
	{
		BT_PROFILE("new m_contactBuffer;");
		m_contactBuffer = new btOpenCLArray<Contact4>(m_context,m_queue,nContacts );
		m_contactBuffer->resize(nContacts);
	}
	

	

	//DeviceUtils::Config dhCfg;
	//Device* deviceHost = DeviceUtils::allocate( TYPE_HOST, dhCfg );
	if( cfg.m_enableParallelSolve )
	{
		

		clFinish(m_queue);
		
		//	contactsIn -> m_contactBuffer
		{
			BT_PROFILE("sortContacts");
			sortContacts( bodyBuf, contactsIn, additionalData, nContacts, cfg );
			clFinish(m_queue);
		}
		
		
		{
			BT_PROFILE("m_copyConstraintKernel");

			

			btInt4 cdata; cdata.x = nContacts;
			btBufferInfoCL bInfo[] = { btBufferInfoCL( m_contactBuffer->getBufferCL() ), btBufferInfoCL( contactsIn->getBufferCL() ) };
//			btLauncherCL launcher( m_queue, data->m_device->getKernel( PATH, "CopyConstraintKernel",  "-I ..\\..\\ -Wf,--c++", 0 ) );
			btLauncherCL launcher( m_queue, m_copyConstraintKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( nContacts, 64 );
			clFinish(m_queue);
		}

		{
			BT_PROFILE("batchContacts");
			Solver::batchContacts( contactsIn, nContacts, m_numConstraints, m_offsets, cfg.m_staticIdx );

		}
	}
	{
			BT_PROFILE("waitForCompletion (batchContacts)");
			clFinish(m_queue);
	}
	
	//================
	
	{
		BT_PROFILE("convertToConstraints");
		Solver::convertToConstraints(  bodyBuf, shapeBuf, contactsIn, contactCOut, additionalData, nContacts, cfg );
	}

	{
		BT_PROFILE("convertToConstraints waitForCompletion");
		clFinish(m_queue);
	}
	
}



void Solver::solveContactConstraint(  const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<RigidBodyBase::Inertia>* shapeBuf, 
			SolverData constraint, void* additionalData, int n )
{
	
	
	btInt4 cdata = btMakeInt4( n, 0, 0, 0 );
	{
		
		const int nn = N_SPLIT*N_SPLIT;

		cdata.x = 0;
		cdata.y = 250;


		int numWorkItems = 64*nn/N_BATCHES;
#ifdef DEBUG_ME
		SolverDebugInfo* debugInfo = new  SolverDebugInfo[numWorkItems];
		adl::btOpenCLArray<SolverDebugInfo> gpuDebugInfo(data->m_device,numWorkItems);
#endif



		{

			BT_PROFILE("m_batchSolveKernel iterations");
			for(int iter=0; iter<m_nIterations; iter++)
			{
				for(int ib=0; ib<N_BATCHES; ib++)
				{
#ifdef DEBUG_ME
					memset(debugInfo,0,sizeof(SolverDebugInfo)*numWorkItems);
					gpuDebugInfo.write(debugInfo,numWorkItems);
#endif


					cdata.z = ib;
					cdata.w = N_SPLIT;

				

					btBufferInfoCL bInfo[] = { 

						btBufferInfoCL( bodyBuf->getBufferCL() ), 
						btBufferInfoCL( shapeBuf->getBufferCL() ), 
						btBufferInfoCL( constraint->getBufferCL() ),
						btBufferInfoCL( m_numConstraints->getBufferCL() ), 
						btBufferInfoCL( m_offsets->getBufferCL() ) 
#ifdef DEBUG_ME
						,	btBufferInfoCL(&gpuDebugInfo)
#endif
						};

					btLauncherCL launcher( m_queue, m_batchSolveKernel );
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
					launcher.setConst(  cdata );
					
					launcher.launch1D( numWorkItems, 64 );

#ifdef DEBUG_ME
					clFinish(m_queue);
					gpuDebugInfo.read(debugInfo,numWorkItems);
					clFinish(m_queue);
					for (int i=0;i<numWorkItems;i++)
					{
						if (debugInfo[i].m_valInt2>0)
						{
							printf("debugInfo[i].m_valInt2 = %d\n",i,debugInfo[i].m_valInt2);
						}

						if (debugInfo[i].m_valInt3>0)
						{
							printf("debugInfo[i].m_valInt3 = %d\n",i,debugInfo[i].m_valInt3);
						}
					}
#endif //DEBUG_ME


				}
			}
		
			clFinish(m_queue);


		}

		cdata.x = 1;
		{
			BT_PROFILE("m_batchSolveKernel iterations2");
			for(int iter=0; iter<m_nIterations; iter++)
			{
				for(int ib=0; ib<N_BATCHES; ib++)
				{
					cdata.z = ib;
					cdata.w = N_SPLIT;

					btBufferInfoCL bInfo[] = { 
						btBufferInfoCL( bodyBuf->getBufferCL() ), 
						btBufferInfoCL( shapeBuf->getBufferCL() ), 
						btBufferInfoCL( constraint->getBufferCL() ),
						btBufferInfoCL( m_numConstraints->getBufferCL() ), 
						btBufferInfoCL( m_offsets->getBufferCL() )
#ifdef DEBUG_ME
						,btBufferInfoCL(&gpuDebugInfo)
#endif //DEBUG_ME
					};
					btLauncherCL launcher( m_queue, m_batchSolveKernel );
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
					launcher.setConst(  cdata );
					launcher.launch1D( 64*nn/N_BATCHES, 64 );
				}
			}
			clFinish(m_queue);
			
		}
#ifdef DEBUG_ME
		delete[] debugInfo;
#endif //DEBUG_ME
	}

	
}

void Solver::convertToConstraints( const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, 
	const btOpenCLArray<RigidBodyBase::Inertia>* shapeBuf, 
	btOpenCLArray<Contact4>* contactsIn, SolverData contactCOut, void* additionalData, 
	int nContacts, const ConstraintCfg& cfg )
{
	btOpenCLArray<Constraint4>* constraintNative =0;

	struct CB
	{
		int m_nContacts;
		float m_dt;
		float m_positionDrift;
		float m_positionConstraintCoeff;
	};

	{
		BT_PROFILE("m_contactToConstraintKernel");
		CB cdata;
		cdata.m_nContacts = nContacts;
		cdata.m_dt = cfg.m_dt;
		cdata.m_positionDrift = cfg.m_positionDrift;
		cdata.m_positionConstraintCoeff = cfg.m_positionConstraintCoeff;

		
		btBufferInfoCL bInfo[] = { btBufferInfoCL( contactsIn->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL() ), btBufferInfoCL( shapeBuf->getBufferCL()),
			btBufferInfoCL( contactCOut->getBufferCL() )};
		btLauncherCL launcher( m_queue, m_contactToConstraintKernel );
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst(  cdata );
		launcher.launch1D( nContacts, 64 );	
		clFinish(m_queue);

	}

}


void Solver::sortContacts(  const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, 
			btOpenCLArray<Contact4>* contactsIn, void* additionalData, 
			int nContacts, const Solver::ConstraintCfg& cfg )
{
	
	

	const int sortAlignment = 512; // todo. get this out of sort
	if( cfg.m_enableParallelSolve )
	{
		

		int sortSize = NEXTMULTIPLEOF( nContacts, sortAlignment );

		btOpenCLArray<u32>* countsNative = m_numConstraints;//BufferUtils::map<TYPE_CL, false>( data->m_device, &countsHost );
		btOpenCLArray<u32>* offsetsNative = m_offsets;//BufferUtils::map<TYPE_CL, false>( data->m_device, &offsetsHost );

		{	//	2. set cell idx
			struct CB
			{
				int m_nContacts;
				int m_staticIdx;
				float m_scale;
				int m_nSplit;
			};

			btAssert( sortSize%64 == 0 );
			CB cdata;
			cdata.m_nContacts = nContacts;
			cdata.m_staticIdx = cfg.m_staticIdx;
			cdata.m_scale = 1.f/(N_OBJ_PER_SPLIT*cfg.m_averageExtent);
			cdata.m_nSplit = N_SPLIT;

			
			btBufferInfoCL bInfo[] = { btBufferInfoCL( contactsIn->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL() ), btBufferInfoCL( m_sortDataBuffer->getBufferCL() ) };
			btLauncherCL launcher( m_queue, m_setSortDataKernel );
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst(  cdata );
			launcher.launch1D( sortSize, 64 );
		}

		{	//	3. sort by cell idx
			int n = N_SPLIT*N_SPLIT;
			int sortBit = 32;
			//if( n <= 0xffff ) sortBit = 16;
			//if( n <= 0xff ) sortBit = 8;
			m_sort32->execute(*m_sortDataBuffer,sortSize);
		}
		{	//	4. find entries
			m_search->execute( *m_sortDataBuffer, nContacts, *countsNative, N_SPLIT*N_SPLIT, btBoundSearchCL::COUNT);

			m_scan->execute( *countsNative, *offsetsNative, N_SPLIT*N_SPLIT );
		}

		{	//	5. sort constraints by cellIdx
			//	todo. preallocate this
//			btAssert( contactsIn->getType() == TYPE_HOST );
//			btOpenCLArray<Contact4>* out = BufferUtils::map<TYPE_CL, false>( data->m_device, contactsIn );	//	copying contacts to this buffer

			{
				

				btInt4 cdata; cdata.x = nContacts;
				btBufferInfoCL bInfo[] = { btBufferInfoCL( contactsIn->getBufferCL() ), btBufferInfoCL( m_contactBuffer->getBufferCL() ), btBufferInfoCL( m_sortDataBuffer->getBufferCL() ) };
				btLauncherCL launcher( m_queue, m_reorderContactKernel );
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
				launcher.setConst(  cdata );
				launcher.launch1D( nContacts, 64 );
			}
//			BufferUtils::unmap<true>( out, contactsIn, nContacts );
		}
	}

	
}



void Solver::batchContacts(  btOpenCLArray<Contact4>* contacts, int nContacts, btOpenCLArray<u32>* nNative, btOpenCLArray<u32>* offsetsNative, int staticIdx )
{

	{
		BT_PROFILE("GPU classTestKernel/Kernel (batch generation?)");
		
		btInt4 cdata;
		cdata.x = nContacts;
		cdata.y = 0;
		cdata.z = staticIdx;

		int numWorkItems = 64*N_SPLIT*N_SPLIT;
#ifdef BATCH_DEBUG
		SolverDebugInfo* debugInfo = new  SolverDebugInfo[numWorkItems];
		adl::btOpenCLArray<SolverDebugInfo> gpuDebugInfo(data->m_device,numWorkItems);
		memset(debugInfo,0,sizeof(SolverDebugInfo)*numWorkItems);
		gpuDebugInfo.write(debugInfo,numWorkItems);
#endif


		btBufferInfoCL bInfo[] = { 
			btBufferInfoCL( contacts->getBufferCL() ), 
			btBufferInfoCL( m_contactBuffer->getBufferCL() ), 
			btBufferInfoCL( nNative->getBufferCL() ), 
			btBufferInfoCL( offsetsNative->getBufferCL() ) 
#ifdef BATCH_DEBUG
			,	btBufferInfoCL(&gpuDebugInfo)
#endif
		};

		
		
		btLauncherCL launcher( m_queue, m_batchingKernel);
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst(  cdata );
		launcher.launch1D( numWorkItems, 64 );
		clFinish(m_queue);

#ifdef BATCH_DEBUG
	aaaa
		Contact4* hostContacts = new Contact4[nContacts];
		m_contactBuffer->read(hostContacts,nContacts);
		clFinish(m_queue);

		gpuDebugInfo.read(debugInfo,numWorkItems);
		clFinish(m_queue);

		for (int i=0;i<numWorkItems;i++)
		{
			if (debugInfo[i].m_valInt1>0)
			{
				printf("catch\n");
			}
			if (debugInfo[i].m_valInt2>0)
			{
				printf("catch22\n");
			}

			if (debugInfo[i].m_valInt3>0)
			{
				printf("catch666\n");
			}

			if (debugInfo[i].m_valInt4>0)
			{
				printf("catch777\n");
			}
		}
		delete[] debugInfo;
#endif //BATCH_DEBUG

	}

//	copy buffer to buffer
	btAssert(m_contactBuffer->size()==nContacts);
	contacts->copyFromOpenCLArray( *m_contactBuffer);
	clFinish(m_queue);//needed?

	
}


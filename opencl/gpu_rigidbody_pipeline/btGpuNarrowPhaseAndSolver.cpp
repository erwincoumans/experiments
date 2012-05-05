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
//Originally written by Erwin Coumans

#include "btGpuNarrowphaseAndSolver.h"

//#include "CustomConvexShape.h"
//#include "CustomConvexPairCollision.h"
#include "LinearMath/btQuickprof.h"

#include "../broadphase_benchmark/btLauncherCL.h"
#include "../broadphase_benchmark/btRadixSort32CL.h"


//#include "BulletDynamics/Dynamics/btRigidBody.h"

#include "../../dynamics/basic_demo/Stubs/AdlMath.h"
#include "../../dynamics/basic_demo/Stubs/AdlContact4.h"
#include "../../dynamics/basic_demo/Stubs/AdlQuaternion.h"
#include "../../dynamics/basic_demo/Stubs/AdlRigidBody.h"


#include "../../dynamics/basic_demo/Stubs/ChNarrowPhase.h"
#include "../../dynamics/basic_demo/Stubs/Solver.h"

int gpuBatchContacts = 1;

int numPairsOut =0;
struct CPUSolveData
{
	u32 m_n[BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT];
	u32 m_offset[BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT];
};


struct ParallelSolveData
{
	btOpenCLArray<u32>* m_numConstraints;
	btOpenCLArray<u32>* m_offsets;
};

struct	CustomDispatchData
{
	btOpenCLArray<ChNarrowphase::ShapeData>* m_ShapeBuffer;
	btAlignedObjectArray<ConvexHeightField*>* m_shapePointers;

	btAlignedObjectArray<int2>* m_pBufPairsCPU;

	btOpenCLArray<int2>* m_convexPairsOutGPU;
	btOpenCLArray<int2>* m_planePairs;

	btOpenCLArray<Contact4>* m_pBufContactOutGPU;
	btAlignedObjectArray<Contact4>* m_pBufContactOutCPU;
	//adl::ChNarrowphase<adl::TYPE_CL>::Data* m_Data;
	ChNarrowphase* m_narrowPhase;


	btAlignedObjectArray<RigidBodyBase::Body>* m_bodyBufferCPU;
	btOpenCLArray<RigidBodyBase::Body>* m_bodyBufferGPU;

	btAlignedObjectArray<RigidBodyBase::Inertia>*	m_inertiaBufferCPU;
	btOpenCLArray<RigidBodyBase::Inertia>*	m_inertiaBufferGPU;

	Solver* m_solverGPU;

	SolverData		m_contactCGPU;
	void*			m_frictionCGPU;

	int m_numAcceleratedShapes;
	int m_numAcceleratedRigidBodies;
};


btGpuNarrowphaseAndSolver::btGpuNarrowphaseAndSolver(cl_context ctx, cl_device_id device, cl_command_queue queue)
	:m_internalData(0) ,m_planeBodyIndex(-1),
	m_context(ctx),
	m_device(device),
	m_queue(queue)
{

	m_internalData = new CustomDispatchData();
	memset(m_internalData,0,sizeof(CustomDispatchData));

	

	m_internalData->m_pBufPairsCPU = new btAlignedObjectArray<int2>;
	m_internalData->m_pBufPairsCPU->resize(MAX_BROADPHASE_COLLISION_CL);
	
	m_internalData->m_convexPairsOutGPU = new btOpenCLArray<int2>(ctx,queue,MAX_BROADPHASE_COLLISION_CL,false);
	m_internalData->m_planePairs = new btOpenCLArray<int2>(ctx,queue,MAX_BROADPHASE_COLLISION_CL,false);
		
	m_internalData->m_pBufContactOutCPU = new btAlignedObjectArray<Contact4>();
	m_internalData->m_pBufContactOutCPU->resize(MAX_BROADPHASE_COLLISION_CL);
	m_internalData->m_bodyBufferCPU = new btAlignedObjectArray<RigidBodyBase::Body>();
	m_internalData->m_bodyBufferCPU->resize(MAX_CONVEX_BODIES_CL);

	m_internalData->m_inertiaBufferCPU = new btAlignedObjectArray<RigidBodyBase::Inertia>();
	m_internalData->m_inertiaBufferCPU->resize(MAX_CONVEX_BODIES_CL);
	
	m_internalData->m_pBufContactOutGPU = new btOpenCLArray<Contact4>(ctx,queue, MAX_BROADPHASE_COLLISION_CL,false);
	m_internalData->m_inertiaBufferGPU = new btOpenCLArray<RigidBodyBase::Inertia>(ctx,queue,MAX_CONVEX_BODIES_CL,false);

	m_internalData->m_solverGPU = new Solver(ctx,device,queue,MAX_BROADPHASE_COLLISION_CL);

	//m_internalData->m_solverDataGPU = adl::Solver<adl::TYPE_CL>::allocate(ctx,queue, MAX_BROADPHASE_COLLISION_CL,false);
	m_internalData->m_bodyBufferGPU = new btOpenCLArray<RigidBodyBase::Body>(ctx,queue, MAX_CONVEX_BODIES_CL,false);
	m_internalData->m_narrowPhase = new ChNarrowphase(ctx,device,queue);

	//m_internalData->m_Data = adl::ChNarrowphase<adl::TYPE_CL>::allocate(m_internalData->m_deviceCL);
//		m_internalData->m_DataCPU = adl::ChNarrowphase<adl::TYPE_HOST>::allocate(m_internalData->m_deviceHost);
		

	m_internalData->m_ShapeBuffer =  new btOpenCLArray<ChNarrowphase::ShapeData>( ctx,queue, MAX_CONVEX_SHAPES_CL,false);	

	m_internalData->m_shapePointers = new btAlignedObjectArray<ConvexHeightField*>();
	m_internalData->m_shapePointers->resize(MAX_CONVEX_SHAPES_CL);

	m_internalData->m_numAcceleratedShapes = 0;
	m_internalData->m_numAcceleratedRigidBodies = 0;

	m_internalData->m_contactCGPU = new btOpenCLArray<Constraint4>(ctx,queue,MAX_BROADPHASE_COLLISION_CL,false);
	//m_internalData->m_frictionCGPU = new btOpenCLArray<adl::Solver<adl::TYPE_CL>::allocateFrictionConstraint( m_internalData->m_deviceCL, MAX_BROADPHASE_COLLISION_CL);

}

int btGpuNarrowphaseAndSolver::registerShape(ConvexHeightField* convexShape)
{
	(*m_internalData->m_shapePointers)[m_internalData->m_numAcceleratedShapes] = convexShape;
	m_internalData->m_narrowPhase->setShape(m_internalData->m_ShapeBuffer, convexShape, m_internalData->m_numAcceleratedShapes, 0.01f);
	return m_internalData->m_numAcceleratedShapes++;
}

cl_mem	btGpuNarrowphaseAndSolver::getBodiesGpu()
{
	return (cl_mem)m_internalData->m_bodyBufferGPU->getBufferCL();
}

cl_mem	btGpuNarrowphaseAndSolver::getBodyInertiasGpu()
{
	return (cl_mem)m_internalData->m_inertiaBufferGPU->getBufferCL();
}


int btGpuNarrowphaseAndSolver::registerRigidBody(int shapeIndex, float mass, const float* position, const float* orientation , bool writeToGpu)
{
	assert(m_internalData->m_numAcceleratedRigidBodies< (MAX_CONVEX_BODIES_CL-1));

	RigidBodyBase::Body& body = m_internalData->m_bodyBufferCPU->at(m_internalData->m_numAcceleratedRigidBodies);

	float friction = 1.f;
	float restitution = 0.f;

	body.m_frictionCoeff = friction;
	body.m_restituitionCoeff = restitution;
	body.m_angVel = make_float4(0.f);
	body.m_linVel = make_float4(0.f);
	body.m_pos = make_float4(position[0],position[1],position[2],0.f);
	body.m_quat = make_float4(orientation[0],orientation[1],orientation[2],orientation[3]);
	body.m_shapeIdx = shapeIndex;
	if (shapeIndex<0)
	{
		body.m_shapeType = CollisionShape::SHAPE_PLANE;
		m_planeBodyIndex = m_internalData->m_numAcceleratedRigidBodies;
	} else
	{
		body.m_shapeType = CollisionShape::SHAPE_CONVEX_HEIGHT_FIELD;
	}
	
	body.m_invMass = mass? 1.f/mass : 0.f;

	if (writeToGpu)
		m_internalData->m_bodyBufferGPU->copyFromHostPointer(&body,1,m_internalData->m_numAcceleratedRigidBodies);

	RigidBodyBase::Inertia& shapeInfo = m_internalData->m_inertiaBufferCPU->at(m_internalData->m_numAcceleratedRigidBodies);

	if (mass==0.f)
	{
		shapeInfo.m_initInvInertia = mtZero();
		shapeInfo.m_invInertia = mtZero();
	} else
	{

		assert(body.m_shapeIdx>=0);

		//approximate using the aabb of the shape

		Aabb aabb = (*m_internalData->m_shapePointers)[shapeIndex]->m_aabb;
		float4 halfExtents = (aabb.m_max - aabb.m_min);

		float4 localInertia;

		float lx=2.f*halfExtents.x;
		float ly=2.f*halfExtents.y;
		float lz=2.f*halfExtents.z;

		localInertia = make_float4( (mass/12.0f) * (ly*ly + lz*lz),
			(mass/12.0f) * (lx*lx + lz*lz),
			(mass/12.0f) * (lx*lx + ly*ly));

		float4 invLocalInertia;
		invLocalInertia.x = 1.f/localInertia.x;
		invLocalInertia.y = 1.f/localInertia.y;
		invLocalInertia.z = 1.f/localInertia.z;
		invLocalInertia.w = 0.f;

		shapeInfo.m_initInvInertia = mtZero();
		shapeInfo.m_initInvInertia.m_row[0].x = invLocalInertia.x;
		shapeInfo.m_initInvInertia.m_row[1].y = invLocalInertia.y;
		shapeInfo.m_initInvInertia.m_row[2].z = invLocalInertia.z;

		Matrix3x3 m = qtGetRotationMatrix( body.m_quat);
		Matrix3x3 mT = mtTranspose( m );
		shapeInfo.m_invInertia = mtMul( mtMul( m, shapeInfo.m_initInvInertia ), mT );

	}

	if (writeToGpu)
		m_internalData->m_inertiaBufferGPU->copyFromHostPointer(&shapeInfo,1,m_internalData->m_numAcceleratedRigidBodies);
	return m_internalData->m_numAcceleratedRigidBodies++;
}

void	btGpuNarrowphaseAndSolver::writeAllBodiesToGpu()
{
	m_internalData->m_bodyBufferGPU->copyFromHostPointer(&m_internalData->m_bodyBufferCPU->at(0),m_internalData->m_numAcceleratedRigidBodies);
	m_internalData->m_inertiaBufferGPU->copyFromHostPointer(&m_internalData->m_inertiaBufferCPU->at(0),m_internalData->m_numAcceleratedRigidBodies);
}



btGpuNarrowphaseAndSolver::~btGpuNarrowphaseAndSolver(void)
{
	if (m_internalData)
	{
		delete m_internalData->m_pBufPairsCPU;
		delete m_internalData->m_convexPairsOutGPU;
		delete m_internalData->m_planePairs;
		delete m_internalData->m_pBufContactOutGPU;
		delete m_internalData->m_inertiaBufferGPU;
		delete m_internalData->m_pBufContactOutCPU;
		delete m_internalData->m_shapePointers;
		delete m_internalData->m_ShapeBuffer;
		delete m_internalData->m_inertiaBufferCPU;
		delete m_internalData->m_contactCGPU;
		delete m_internalData->m_bodyBufferGPU;
		delete m_internalData->m_solverGPU;
		delete m_internalData->m_bodyBufferCPU;
		delete m_internalData->m_narrowPhase;

		delete m_internalData;

	}

}





void btGpuNarrowphaseAndSolver::computeContactsAndSolver(cl_mem broadphasePairs, int numBroadphasePairs) 
{


	BT_PROFILE("computeContactsAndSolver");
	bool bGPU = (m_internalData != 0);
	int maxBodyIndex = m_internalData->m_numAcceleratedRigidBodies;

	if (!maxBodyIndex)
		return;
	int numOfConvexRBodies = maxBodyIndex;

	ChNarrowphaseBase::Config cfgNP;
	cfgNP.m_collisionMargin = 0.01f;
	int nContactOut = 0;
	//printf("convexPairsOut.m_size = %d\n",m_internalData->m_convexPairsOutGPU->m_size);


	btOpenCLArray<int2> broadphasePairsGPU(m_context,m_queue);
	broadphasePairsGPU.setFromOpenCLBuffer(broadphasePairs,numBroadphasePairs);

	bool useCulling = true;
	if (useCulling)
	{
		BT_PROFILE("ChNarrowphase::culling");
		clFinish(m_queue);

		numPairsOut = m_internalData->m_narrowPhase->culling(
			&broadphasePairsGPU, 
			numBroadphasePairs,
			m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer,
			m_internalData->m_convexPairsOutGPU,
			cfgNP);
	}

	{
		BT_PROFILE("ChNarrowphase::execute");
		if (useCulling)
		{
		
			if (m_planeBodyIndex>=0)
			{
				BT_PROFILE("ChNarrowphase:: plane versus convex");
				//todo: get rid of this dynamic allocation
				int2* hostPairs = new int2[m_internalData->m_numAcceleratedRigidBodies-1];
				int index=0;
				for (int i=0;i<m_internalData->m_numAcceleratedRigidBodies;i++)
				{
					if (i!=m_planeBodyIndex)
					{
						hostPairs[index].x = m_planeBodyIndex;
						hostPairs[index].y = i;
						index++;
					}
				}
				assert(m_internalData->m_numAcceleratedRigidBodies-1 == index);
				m_internalData->m_planePairs->copyFromHostPointer(hostPairs,index);
				clFinish(m_queue);

				delete[]hostPairs;
				//convex versus plane
				m_internalData->m_narrowPhase->execute(m_internalData->m_planePairs, index, m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer, 
					0,0,m_internalData->m_pBufContactOutGPU, nContactOut, cfgNP);
			}
		
			//convex versus convex
			m_internalData->m_narrowPhase->execute(m_internalData->m_convexPairsOutGPU,numPairsOut, m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer, m_internalData->m_pBufContactOutGPU, nContactOut, cfgNP);
		} else
		{
			m_internalData->m_narrowPhase->execute(&broadphasePairsGPU, numBroadphasePairs, m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer, m_internalData->m_pBufContactOutGPU, nContactOut, cfgNP);
		}

		clFinish(m_queue);
	}
	
	if (!nContactOut)
		return;
	

	bool useSolver = true;//true;//false;

	if (useSolver)
	{
		float dt=1./60.;
		SolverBase::ConstraintCfg csCfg( dt );
		csCfg.m_enableParallelSolve = true;
		csCfg.m_averageExtent = 0.2f;//@TODO m_averageObjExtent;
		csCfg.m_staticIdx = m_planeBodyIndex;

		btOpenCLArray<Contact4>* contactsIn = m_internalData->m_pBufContactOutGPU;
		const btOpenCLArray<RigidBodyBase::Body>* bodyBuf = m_internalData->m_bodyBufferGPU;
		void* additionalData = m_internalData->m_frictionCGPU;
		const btOpenCLArray<RigidBodyBase::Inertia>* shapeBuf = m_internalData->m_inertiaBufferGPU;
		SolverData contactCOut = m_internalData->m_contactCGPU;
		int nContacts = nContactOut;

		bool useCPU=false;

		{
			BT_PROFILE("GPU batch");

			{
				//@todo: just reserve it, without copy of original contact (unless we use warmstarting)
				if( m_internalData->m_solverGPU->m_contactBuffer)
				{
					m_internalData->m_solverGPU->m_contactBuffer->resize(nContacts);
				}

				if( m_internalData->m_solverGPU->m_contactBuffer == 0 )
				{
					m_internalData->m_solverGPU->m_contactBuffer = new btOpenCLArray<Contact4>(m_context,m_queue, nContacts );
					m_internalData->m_solverGPU->m_contactBuffer->resize(nContacts);
				}

				btOpenCLArray<Contact4>* contactNative  = contactsIn;
				const btOpenCLArray<RigidBodyBase::Body>* bodyNative = bodyBuf;
				

				{
					
					//btOpenCLArray<RigidBodyBase::Body>* bodyNative = btOpenCLArrayUtils::map<adl::TYPE_CL, true>( data->m_device, bodyBuf );
					//btOpenCLArray<Contact4>* contactNative = btOpenCLArrayUtils::map<adl::TYPE_CL, true>( data->m_device, contactsIn );

					const int sortAlignment = 512; // todo. get this out of sort
					if( csCfg.m_enableParallelSolve )
					{
						

						int sortSize = NEXTMULTIPLEOF( nContacts, sortAlignment );

						btOpenCLArray<u32>* countsNative = m_internalData->m_solverGPU->m_numConstraints;
						btOpenCLArray<u32>* offsetsNative = m_internalData->m_solverGPU->m_offsets;

						{	//	2. set cell idx
							BT_PROFILE("GPU set cell idx");
							struct CB
							{
								int m_nContacts;
								int m_staticIdx;
								float m_scale;
								int m_nSplit;
							};

							ADLASSERT( sortSize%64 == 0 );
							CB cdata;
							cdata.m_nContacts = nContacts;
							cdata.m_staticIdx = csCfg.m_staticIdx;
							cdata.m_scale = 1.f/(BT_SOLVER_N_OBJ_PER_SPLIT*csCfg.m_averageExtent);
							cdata.m_nSplit = BT_SOLVER_N_SPLIT;

							m_internalData->m_solverGPU->m_sortDataBuffer->resize(nContacts);

							
							btBufferInfoCL bInfo[] = { btBufferInfoCL( contactNative->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL()), btBufferInfoCL( m_internalData->m_solverGPU->m_sortDataBuffer->getBufferCL()) };
							btLauncherCL launcher(m_queue, m_internalData->m_solverGPU->m_setSortDataKernel );
							launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
							launcher.setConst( cdata );
							launcher.launch1D( sortSize, 64 );
						}
						bool gpuRadixSort=true;
						if (gpuRadixSort)
						{	//	3. sort by cell idx
							BT_PROFILE("gpuRadixSort");
							int n = BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT;
							int sortBit = 32;
							//if( n <= 0xffff ) sortBit = 16;
							//if( n <= 0xff ) sortBit = 8;
							//adl::RadixSort<adl::TYPE_CL>::execute( data->m_sort, *data->m_sortDataBuffer, sortSize );
							//adl::RadixSort32<adl::TYPE_CL>::execute( data->m_sort32, *data->m_sortDataBuffer, sortSize );
							btOpenCLArray<btSortData>& keyValuesInOut = *(m_internalData->m_solverGPU->m_sortDataBuffer);
							this->m_internalData->m_solverGPU->m_sort32->execute(keyValuesInOut);

							/*btAlignedObjectArray<btSortData> hostValues;
							keyValuesInOut.copyToHost(hostValues);
							printf("hostValues.size=%d\n",hostValues.size());
							*/

						}



						
						{	
							//	4. find entries
							BT_PROFILE("gpuBoundSearch");
							
							m_internalData->m_solverGPU->m_search->execute(*m_internalData->m_solverGPU->m_sortDataBuffer,nContacts,*countsNative,
								BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT,btBoundSearchCL::COUNT);

							
							//adl::BoundSearch<adl::TYPE_CL>::execute( data->m_search, *data->m_sortDataBuffer, nContacts, *countsNative, 
							//	BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT, adl::BoundSearchBase::COUNT );

							//unsigned int sum;
							m_internalData->m_solverGPU->m_scan->execute(*countsNative,*offsetsNative, BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT);//,&sum );
							//printf("sum = %d\n",sum);
						} 


						{	//	5. sort constraints by cellIdx
							{
								BT_PROFILE("gpu m_reorderContactKernel");
							
								btInt4 cdata; 
								cdata.x = nContacts;

								btBufferInfoCL bInfo[] = { btBufferInfoCL( contactNative->getBufferCL() ), btBufferInfoCL( m_internalData->m_solverGPU->m_contactBuffer->getBufferCL())
									, btBufferInfoCL( m_internalData->m_solverGPU->m_sortDataBuffer->getBufferCL()) };
								btLauncherCL launcher(m_queue,m_internalData->m_solverGPU->m_reorderContactKernel);
								launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
								launcher.setConst( cdata );
								launcher.launch1D( nContacts, 64 );
							}
						}

					}

				}

				clFinish(m_queue);

				{
					BT_PROFILE("gpu m_copyConstraintKernel");
					
					btInt4 cdata; cdata.x = nContacts;
					btBufferInfoCL bInfo[] = { btBufferInfoCL(  m_internalData->m_solverGPU->m_contactBuffer->getBufferCL() ), btBufferInfoCL( contactNative->getBufferCL() ) };
					btLauncherCL launcher(m_queue, m_internalData->m_solverGPU->m_copyConstraintKernel );
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
					launcher.setConst(  cdata );
					launcher.launch1D( nContacts, 64 );
					clFinish(m_queue);
				}
					
				bool compareGPU = false;
				if (gpuBatchContacts)
				{
					BT_PROFILE("gpu batchContacts");
					m_internalData->m_solverGPU->batchContacts( contactNative, nContacts, m_internalData->m_solverGPU->m_numConstraints, m_internalData->m_solverGPU->m_offsets, csCfg.m_staticIdx );
				}

				if (1)
				{
					BT_PROFILE("gpu convertToConstraints");
					m_internalData->m_solverGPU->convertToConstraints( bodyBuf, shapeBuf, contactNative, contactCOut, additionalData, nContacts, csCfg );
					clFinish(m_queue);
				}

			}
		}


		if (1)
		{
			BT_PROFILE("GPU solveContactConstraint");
			m_internalData->m_solverGPU->m_nIterations = 4;//10
			m_internalData->m_solverGPU->solveContactConstraint(m_internalData->m_bodyBufferGPU, 
				m_internalData->m_inertiaBufferGPU, 
				m_internalData->m_contactCGPU,
				0, 
				nContactOut );

			clFinish(m_queue);
		}


#if 0
		if (0)
		{
			BT_PROFILE("read body velocities back to CPU");
			//read body updated linear/angular velocities back to CPU
			m_internalData->m_bodyBufferGPU->read(
				m_internalData->m_bodyBufferCPU->m_ptr,numOfConvexRBodies);
			adl::DeviceUtils::waitForCompletion( m_internalData->m_deviceCL );
		}
#endif

	}

}

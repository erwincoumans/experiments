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

//#define USE_CONVEX_CONVEX_HOST 1
//#define DISABLE_CONVEX_HEIGHTFIELD

bool useConvexHeightfield = false;

#include "btGpuNarrowphaseAndSolver.h"
#include "../rendering/WavefrontObjLoader/objLoader.h"
//#include "CustomConvexShape.h"
//#include "CustomConvexPairCollision.h"
#include "LinearMath/btQuickprof.h"

#include "../broadphase_benchmark/btLauncherCL.h"
#include "../broadphase_benchmark/btRadixSort32CL.h"

#include "btConvexUtility.h"
#include "../gpu_rigidbody_pipeline2/ConvexHullContact.h"


//#include "BulletDynamics/Dynamics/btRigidBody.h"

#include "../../dynamics/basic_demo/Stubs/AdlMath.h"
#include "../../dynamics/basic_demo/Stubs/AdlContact4.h"
#include "../../dynamics/basic_demo/Stubs/AdlQuaternion.h"
#include "../../dynamics/basic_demo/Stubs/AdlRigidBody.h"


#include "../../dynamics/basic_demo/Stubs/ChNarrowPhase.h"
#include "../../dynamics/basic_demo/Stubs/Solver.h"


#define MAX_CONVEX_UNIQUE_EDGES 8192
#define MAX_CONVEX_VERTICES 8192
#define MAX_CONVEX_INDICES 8192

int gpuBatchContacts = 1;

int numPairsOut =0;
int numPairsTotal =0;

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
#ifndef DISABLE_CONVEX_HEIGHTFIELD
	btAlignedObjectArray<ConvexHeightField*>* m_shapePointers;
#endif //DISABLE_CONVEX_HEIGHTFIELD
    
	btAlignedObjectArray<btConvexUtility*>* m_convexData;
    
	btAlignedObjectArray<ConvexPolyhedronCL> m_convexPolyhedra;
	btAlignedObjectArray<btVector3> m_uniqueEdges;
	btAlignedObjectArray<btVector3> m_convexVertices;
	btAlignedObjectArray<int> m_convexIndices;
    
	btOpenCLArray<ConvexPolyhedronCL>* m_convexPolyhedraGPU;
	btOpenCLArray<btVector3>* m_uniqueEdgesGPU;
	btOpenCLArray<btVector3>* m_convexVerticesGPU;
	btOpenCLArray<int>* m_convexIndicesGPU;
    
    
	btAlignedObjectArray<btGpuFace> m_convexFaces;
	btOpenCLArray<btGpuFace>* m_convexFacesGPU;
    
	GpuSatCollision*	m_gpuSatCollision;
	
    
    
    
    
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
    
	btAlignedObjectArray<btCollidable>	m_collidablesCPU;
	btOpenCLArray<btCollidable>*	m_collidablesGPU;
    
    
};


btGpuNarrowphaseAndSolver::btGpuNarrowphaseAndSolver(cl_context ctx, cl_device_id device, cl_command_queue queue)
:m_internalData(0) ,m_planeBodyIndex(-1),m_static0Index(-1),
m_context(ctx),
m_device(device),
m_queue(queue)
{
    
	m_internalData = new CustomDispatchData();
	memset(m_internalData,0,sizeof(CustomDispatchData));
    
	
	m_internalData->m_gpuSatCollision = new GpuSatCollision(ctx,device,queue);
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
	m_internalData->m_collidablesGPU = new btOpenCLArray<btCollidable>(ctx,queue,MAX_CONVEX_SHAPES_CL);
    
	m_internalData->m_solverGPU = new Solver(ctx,device,queue,MAX_BROADPHASE_COLLISION_CL);
    
	//m_internalData->m_solverDataGPU = adl::Solver<adl::TYPE_CL>::allocate(ctx,queue, MAX_BROADPHASE_COLLISION_CL,false);
	m_internalData->m_bodyBufferGPU = new btOpenCLArray<RigidBodyBase::Body>(ctx,queue, MAX_CONVEX_BODIES_CL,false);
	m_internalData->m_narrowPhase = new ChNarrowphase(ctx,device,queue);
    
	m_internalData->m_convexFacesGPU = new btOpenCLArray<btGpuFace>(ctx,queue,MAX_CONVEX_SHAPES_CL*MAX_FACES_PER_SHAPE,false);
    
	m_internalData->m_convexPolyhedraGPU = new btOpenCLArray<ConvexPolyhedronCL>(ctx,queue,MAX_CONVEX_SHAPES_CL,false);
	m_internalData->m_uniqueEdgesGPU = new btOpenCLArray<btVector3>(ctx,queue,MAX_CONVEX_UNIQUE_EDGES,true);
	m_internalData->m_convexVerticesGPU = new btOpenCLArray<btVector3>(ctx,queue,MAX_CONVEX_VERTICES,true);
	m_internalData->m_convexIndicesGPU = new btOpenCLArray<int>(ctx,queue,MAX_CONVEX_INDICES,true);
    
    
	//m_internalData->m_Data = adl::ChNarrowphase<adl::TYPE_CL>::allocate(m_internalData->m_deviceCL);
    //		m_internalData->m_DataCPU = adl::ChNarrowphase<adl::TYPE_HOST>::allocate(m_internalData->m_deviceHost);
    
    
	m_internalData->m_ShapeBuffer =  new btOpenCLArray<ChNarrowphase::ShapeData>( ctx,queue, MAX_CONVEX_SHAPES_CL,false);
    
#ifndef DISABLE_CONVEX_HEIGHTFIELD
	m_internalData->m_shapePointers = new btAlignedObjectArray<ConvexHeightField*>();
#endif //DISABLE_CONVEX_HEIGHTFIELD
    
	m_internalData->m_convexData = new btAlignedObjectArray<btConvexUtility* >();
    
#ifndef DISABLE_CONVEX_HEIGHTFIELD
	m_internalData->m_shapePointers->resize(MAX_CONVEX_SHAPES_CL);
#endif
	m_internalData->m_convexData->resize(MAX_CONVEX_SHAPES_CL);
	m_internalData->m_convexPolyhedra.resize(MAX_CONVEX_SHAPES_CL);
    
	m_internalData->m_numAcceleratedShapes = 0;
	m_internalData->m_numAcceleratedRigidBodies = 0;
    
	m_internalData->m_contactCGPU = new btOpenCLArray<Constraint4>(ctx,queue,MAX_BROADPHASE_COLLISION_CL,false);
	//m_internalData->m_frictionCGPU = new btOpenCLArray<adl::Solver<adl::TYPE_CL>::allocateFrictionConstraint( m_internalData->m_deviceCL, MAX_BROADPHASE_COLLISION_CL);
    
}

int btGpuNarrowphaseAndSolver::allocateCollidable()
{
	int curSize = m_internalData->m_collidablesCPU.size();
	m_internalData->m_collidablesCPU.expand();
	return curSize;
}

int btGpuNarrowphaseAndSolver::registerConcaveMeshShape(class objLoader* obj,btCollidable& col)
{
	m_concaveMeshes.push_back(obj);
	return m_concaveMeshes.size()-1;
}


int btGpuNarrowphaseAndSolver::registerConvexHullShape(ConvexHeightField* convexShape,btConvexUtility* convexPtr,btCollidable& col)
{
#ifndef DISABLE_CONVEX_HEIGHTFIELD
	m_internalData->m_shapePointers->resize(m_internalData->m_numAcceleratedShapes+1);
#endif
	m_internalData->m_convexData->resize(m_internalData->m_numAcceleratedShapes+1);
	m_internalData->m_ShapeBuffer->resize(m_internalData->m_numAcceleratedShapes+1);
	m_internalData->m_convexPolyhedra.resize(m_internalData->m_numAcceleratedShapes+1);
	
    
	ConvexPolyhedronCL& convex = m_internalData->m_convexPolyhedra.at(m_internalData->m_convexPolyhedra.size()-1);
	convex.mC = convexPtr->mC;
	convex.mE = convexPtr->mE;
	convex.m_extents= convexPtr->m_extents;
	convex.m_localCenter = convexPtr->m_localCenter;
	convex.m_radius = convexPtr->m_radius;
	
	convex.m_numUniqueEdges = convexPtr->m_uniqueEdges.size();
	int edgeOffset = m_internalData->m_uniqueEdges.size();
	convex.m_uniqueEdgesOffset = edgeOffset;
	
	m_internalData->m_uniqueEdges.resize(edgeOffset+convex.m_numUniqueEdges);
    
	//convex data here
	int i;
	for ( i=0;i<convexPtr->m_uniqueEdges.size();i++)
	{
		m_internalData->m_uniqueEdges[edgeOffset+i] = convexPtr->m_uniqueEdges[i];
	}
    
	int faceOffset = m_internalData->m_convexFaces.size();
	convex.m_faceOffset = faceOffset;
	convex.m_numFaces = convexPtr->m_faces.size();
	m_internalData->m_convexFaces.resize(faceOffset+convex.m_numFaces);
	for (i=0;i<convexPtr->m_faces.size();i++)
	{
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_plane.x = convexPtr->m_faces[i].m_plane[0];
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_plane.y = convexPtr->m_faces[i].m_plane[1];
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_plane.z = convexPtr->m_faces[i].m_plane[2];
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_plane.w = convexPtr->m_faces[i].m_plane[3];
		int indexOffset = m_internalData->m_convexIndices.size();
		int numIndices = convexPtr->m_faces[i].m_indices.size();
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_numIndices = numIndices;
		m_internalData->m_convexFaces[convex.m_faceOffset+i].m_indexOffset = indexOffset;
		m_internalData->m_convexIndices.resize(indexOffset+numIndices);
		for (int p=0;p<numIndices;p++)
		{
			m_internalData->m_convexIndices[indexOffset+p] = convexPtr->m_faces[i].m_indices[p];
		}
	}
    
	convex.m_numVertices = convexPtr->m_vertices.size();
	int vertexOffset = m_internalData->m_convexVertices.size();
	convex.m_vertexOffset =vertexOffset;
	m_internalData->m_convexVertices.resize(vertexOffset+convex.m_numVertices);
	for (int i=0;i<convexPtr->m_vertices.size();i++)
	{
		m_internalData->m_convexVertices[vertexOffset+i] = convexPtr->m_vertices[i];
	}
#ifndef DISABLE_CONVEX_HEIGHTFIELD
	(*m_internalData->m_shapePointers)[m_internalData->m_numAcceleratedShapes] = convexShape;
#endif
	(*m_internalData->m_convexData)[m_internalData->m_numAcceleratedShapes] = convexPtr;
	m_internalData->m_narrowPhase->setShape(m_internalData->m_ShapeBuffer, convexShape, m_internalData->m_numAcceleratedShapes, 0.01f);
	
	m_internalData->m_convexFacesGPU->copyFromHost(m_internalData->m_convexFaces);
    
	m_internalData->m_convexPolyhedraGPU->copyFromHost(m_internalData->m_convexPolyhedra);
	m_internalData->m_uniqueEdgesGPU->copyFromHost(m_internalData->m_uniqueEdges);
	m_internalData->m_convexVerticesGPU->copyFromHost(m_internalData->m_convexVertices);
	m_internalData->m_convexIndicesGPU->copyFromHost(m_internalData->m_convexIndices);
    
    
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

btCollidable& btGpuNarrowphaseAndSolver::getCollidableCpu(int collidableIndex)
{
	return m_internalData->m_collidablesCPU[collidableIndex];
}

const btCollidable& btGpuNarrowphaseAndSolver::getCollidableCpu(int collidableIndex) const
{
	return m_internalData->m_collidablesGPU->at(collidableIndex);
}


void btGpuNarrowphaseAndSolver::setObjectTransform(const float* position, const float* orientation , int bodyIndex)
{
	RigidBodyBase::Body* body = &m_internalData->m_bodyBufferCPU->at(bodyIndex);
	m_internalData->m_bodyBufferGPU->copyToHostPointer(body,1,bodyIndex);
    
	body->m_pos.x = position[0];
	body->m_pos.y = position[1];
	body->m_pos.z = position[2];
	body->m_quat.x = orientation[0];
	body->m_quat.y = orientation[1];
	body->m_quat.z = orientation[2];
	body->m_quat.w = orientation[3];
	m_internalData->m_bodyBufferGPU->copyFromHostPointer(body,1,bodyIndex);
    
}


int btGpuNarrowphaseAndSolver::registerRigidBody(int collidableIndex, float mass, const float* position, const float* orientation , const float* aabbMinPtr, const float* aabbMaxPtr,bool writeToGpu)
{
	const float4 aabbMin = *(float4*)aabbMinPtr;
	const float4 aabbMax = *(float4*)aabbMaxPtr;
	
	assert(m_internalData->m_numAcceleratedRigidBodies< (MAX_CONVEX_BODIES_CL-1));
    
	m_internalData->m_bodyBufferGPU->resize(m_internalData->m_numAcceleratedRigidBodies+1);
    
	RigidBodyBase::Body& body = m_internalData->m_bodyBufferCPU->at(m_internalData->m_numAcceleratedRigidBodies);
    
	float friction = 1.f;
	float restitution = 0.f;
    
	body.m_frictionCoeff = friction;
	body.m_restituitionCoeff = restitution;
	body.m_angVel = make_float4(0.f);
	body.m_linVel = make_float4(0.f);
	body.m_pos = make_float4(position[0],position[1],position[2],0.f);
	body.m_quat = make_float4(orientation[0],orientation[1],orientation[2],orientation[3]);
	body.m_collidableIdx = collidableIndex;
	body.m_shapeType = m_internalData->m_collidablesCPU.at(collidableIndex).m_shapeType;
	//body.m_shapeType = shapeType;
	/*if (shapeType==CollisionShape::SHAPE_PLANE)
     {
     m_planeBodyIndex = m_internalData->m_numAcceleratedRigidBodies;
     }
     */
    
	
	body.m_invMass = mass? 1.f/mass : 0.f;
    
	if (writeToGpu)
	{
		m_internalData->m_bodyBufferGPU->copyFromHostPointer(&body,1,m_internalData->m_numAcceleratedRigidBodies);
	}
    
	RigidBodyBase::Inertia& shapeInfo = m_internalData->m_inertiaBufferCPU->at(m_internalData->m_numAcceleratedRigidBodies);
    
	if (mass==0.f)
	{
		if (m_internalData->m_numAcceleratedRigidBodies==0)
			m_static0Index = 0;
        
		shapeInfo.m_initInvInertia = mtZero();
		shapeInfo.m_invInertia = mtZero();
	} else
	{
        
		assert(body.m_collidableIdx>=0);
        
		//approximate using the aabb of the shape
        
		//Aabb aabb = (*m_internalData->m_shapePointers)[shapeIndex]->m_aabb;
		float4 halfExtents = (aabbMax-aabbMin);//*0.5f;//fake larger inertia makes demos more stable ;-)
        
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
	m_internalData->m_bodyBufferGPU->resize(m_internalData->m_numAcceleratedRigidBodies);
	m_internalData->m_inertiaBufferGPU->resize(m_internalData->m_numAcceleratedRigidBodies);
    
	m_internalData->m_bodyBufferGPU->copyFromHostPointer(&m_internalData->m_bodyBufferCPU->at(0),m_internalData->m_numAcceleratedRigidBodies);
	m_internalData->m_inertiaBufferGPU->copyFromHostPointer(&m_internalData->m_inertiaBufferCPU->at(0),m_internalData->m_numAcceleratedRigidBodies);
    
	m_internalData->m_collidablesGPU->copyFromHost(m_internalData->m_collidablesCPU);
    
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
		delete m_internalData->m_collidablesGPU;
		delete m_internalData->m_pBufContactOutCPU;
#ifndef DISABLE_CONVEX_HEIGHTFIELD
		delete m_internalData->m_shapePointers;
#endif
		delete m_internalData->m_convexData;
		delete m_internalData->m_ShapeBuffer;
		delete m_internalData->m_inertiaBufferCPU;
		delete m_internalData->m_contactCGPU;
		delete m_internalData->m_bodyBufferGPU;
		delete m_internalData->m_solverGPU;
		delete m_internalData->m_bodyBufferCPU;
		delete m_internalData->m_narrowPhase;
		delete m_internalData->m_gpuSatCollision;
		delete m_internalData->m_convexFacesGPU;
		delete m_internalData->m_convexPolyhedraGPU;
		delete m_internalData->m_uniqueEdgesGPU;
		delete m_internalData->m_convexVerticesGPU;
		delete m_internalData->m_convexIndicesGPU;
		delete m_internalData;
        
	}
    
}

static bool sortfnc(const btSortData& a,const btSortData& b)
{
	return (a.m_key<b.m_key);
}

btAlignedObjectArray<u32> idxBuffer;
btAlignedObjectArray<btSortData> sortData;
btAlignedObjectArray<Contact4> old;

static
__inline
int sortConstraintByBatch( Contact4* cs, int n, int simdWidth , CustomDispatchData*	internalData)
{
	int numIter = 0;
    
	sortData.resize(n);
	idxBuffer.resize(n);
	old.resize(n);
	
	u32* idxSrc = &idxBuffer[0];
	u32* idxDst = &idxBuffer[0];
	int nIdxSrc, nIdxDst;
    
	const int N_FLG = 256;
	const int FLG_MASK = N_FLG-1;
	u32 flg[N_FLG/32];
#if defined(_DEBUG)
	for(int i=0; i<n; i++)
		cs[i].getBatchIdx() = -1;
#endif
	for(int i=0; i<n; i++) idxSrc[i] = i;
	nIdxSrc = n;
    
	int batchIdx = 0;
    
	{
		BT_PROFILE("cpu batch innerloop");
		while( nIdxSrc )
		{
			numIter++;
			nIdxDst = 0;
			int nCurrentBatch = 0;
            
			//	clear flag
			for(int i=0; i<N_FLG/32; i++) flg[i] = 0;
            
			for(int i=0; i<nIdxSrc; i++)
			{
				int idx = idxSrc[i];
				btAssert( idx < n );
				//	check if it can go
				int bodyAS = cs[idx].m_bodyAPtrAndSignBit;
				int bodyBS = cs[idx].m_bodyBPtrAndSignBit;
                
				if (bodyAS<0)
					printf("A static\n");
                
				if (bodyBS<0)
					printf("B static\n");
                
				int bodyA = abs(bodyAS);
				int bodyB = abs(bodyBS);
                
				int aIdx = bodyA & FLG_MASK;
				int bIdx = bodyB & FLG_MASK;
                
				u32 aUnavailable = flg[ aIdx/32 ] & (1<<(aIdx&31));
				u32 bUnavailable = flg[ bIdx/32 ] & (1<<(bIdx&31));
                
                //use inv_mass!
				aUnavailable = (bodyAS>=0)? aUnavailable:0;//
				bUnavailable = (bodyBS>=0)? bUnavailable:0;
                
				if( aUnavailable==0 && bUnavailable==0 ) // ok
				{
					flg[ aIdx/32 ] |= (1<<(aIdx&31));
					flg[ bIdx/32 ] |= (1<<(bIdx&31));
					cs[idx].getBatchIdx() = batchIdx;
					sortData[idx].m_key = batchIdx;
					sortData[idx].m_value = idx;
                    
					{
						nCurrentBatch++;
						if( nCurrentBatch == simdWidth )
						{
							nCurrentBatch = 0;
							for(int i=0; i<N_FLG/32; i++) flg[i] = 0;
						}
					}
				}
				else
				{
					idxDst[nIdxDst++] = idx;
				}
			}
			swap2( idxSrc, idxDst );
			swap2( nIdxSrc, nIdxDst );
			batchIdx ++;
		}
	}
	{
		BT_PROFILE("quickSort");
		sortData.quickSort(sortfnc);
	}
	
	
	{
        BT_PROFILE("reorder");
		//	reorder
		
		memcpy( &old[0], cs, sizeof(Contact4)*n);
		for(int i=0; i<n; i++)
		{
			int idx = sortData[i].m_value;
			cs[i] = old[idx];
		}
	}
    
	
#if defined(_DEBUG)
    //		debugPrintf( "nBatches: %d\n", batchIdx );
	for(int i=0; i<n; i++)
    {
        btAssert( cs[i].getBatchIdx() != -1 );
    }
#endif
	return batchIdx;
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
    
	bool useCulling = false;
	if (useConvexHeightfield)
		useCulling = true;
    
	clFinish(m_queue);
	numPairsTotal = numBroadphasePairs;
	numPairsOut = numBroadphasePairs;
	if (useCulling)
	{
		BT_PROFILE("ChNarrowphase::culling");
		
        
		
		numPairsOut = m_internalData->m_narrowPhase->culling(
                                                             &broadphasePairsGPU,
                                                             numBroadphasePairs,
                                                             m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer,
                                                             m_internalData->m_convexPairsOutGPU,
                                                             cfgNP);
	}
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
	}
    
    
	
	{
		BT_PROFILE("ChNarrowphase::execute");
		
		{
			//convex versus convex
			//m_internalData->m_narrowPhase->execute(m_internalData->m_convexPairsOutGPU,numPairsOut, m_internalData->m_bodyBufferGPU, m_internalData->m_ShapeBuffer, m_internalData->m_pBufContactOutGPU, nContactOut, cfgNP);
            
			if (!useConvexHeightfield)
			{
                
				m_internalData->m_pBufContactOutGPU->resize(nContactOut);
                
				if (useCulling)
				{
					m_internalData->m_convexPairsOutGPU->resize(numPairsOut);
					
					m_internalData->m_gpuSatCollision->computeConvexConvexContactsGPUSAT(
                                                                                         m_internalData->m_convexPairsOutGPU,numPairsOut,
                                                                                         m_internalData->m_bodyBufferGPU,
                                                                                         m_internalData->m_ShapeBuffer,
                                                                                         m_internalData->m_pBufContactOutGPU,
                                                                                         nContactOut, cfgNP, *m_internalData->m_convexPolyhedraGPU,*m_internalData->m_convexVerticesGPU,*m_internalData->m_uniqueEdgesGPU,
                                                                                         *m_internalData->m_convexFacesGPU,*m_internalData->m_convexIndicesGPU,*m_internalData->m_collidablesGPU);
				} else
				{
					m_internalData->m_convexPairsOutGPU->resize(numBroadphasePairs);
                    
					//convex versus concave convex
                    
					m_internalData->m_gpuSatCollision->computeConvexConvexContactsGPUSAT(
                                                                                         &broadphasePairsGPU, numBroadphasePairs,
                                                                                         m_internalData->m_bodyBufferGPU,
                                                                                         m_internalData->m_ShapeBuffer,
                                                                                         m_internalData->m_pBufContactOutGPU,
                                                                                         nContactOut, cfgNP, *m_internalData->m_convexPolyhedraGPU,
                                                                                         *m_internalData->m_convexVerticesGPU,
                                                                                         *m_internalData->m_uniqueEdgesGPU,
                                                                                         *m_internalData->m_convexFacesGPU,
                                                                                         *m_internalData->m_convexIndicesGPU,
                                                                                         
                                                                                         *m_internalData->m_collidablesGPU);
                    
					//convex versus concave trimesh
                    
					if (broadphasePairsGPU.size())
					{
						BT_PROFILE("concave-convex");
						btAlignedObjectArray<int2> hostPairs;
						broadphasePairsGPU.copyToHost(hostPairs);
						m_internalData->m_bodyBufferGPU->copyToHost(*m_internalData->m_bodyBufferCPU);
                        
						//concave case...
						//for now, compute overlapping triangles and run convex-convex again with those pairs
						for (int i=0;i<numBroadphasePairs;i++)
						{
							int2 pair = hostPairs[i];
							RigidBodyBase::Body& bodyA = m_internalData->m_bodyBufferCPU->at(pair.x);
							RigidBodyBase::Body& bodyB = m_internalData->m_bodyBufferCPU->at(pair.y);
                            
							if (bodyA.m_shapeType == CollisionShape::SHAPE_CONCAVE_TRIMESH)
							{
								//bodyA.m_shapeIdx =
                                
								printf("concave x\n");
                                
								int collidableA = bodyA.m_collidableIdx;
								int collidableIndexB = bodyB.m_collidableIdx;
								int shapeA = m_internalData->m_collidablesCPU[collidableA].m_shapeIndex;
								int shapeB = m_internalData->m_collidablesCPU[collidableIndexB].m_shapeIndex;
								objLoader* obj = m_concaveMeshes[shapeA];
                                
                                
                                
								for (int f=0;f<obj->faceCount;f++)
								{
									obj_face* face = obj->faceList[f];
                                    //for now we ignore quads, only tris are allowed
                                    btAssert(face->vertex_count==3);
                                    
                                    if (face->vertex_count==3)
                                    {
                                        int bodyIndexA = pair.x;
                                        int bodyIndexB = pair.y;
                                        btAlignedObjectArray<btCollidable> hostCollidablesA;
                                        btAlignedObjectArray<ConvexPolyhedronCL> hostConvexDataA;
                                        int collidableIndexA = 0;
                                        btAlignedObjectArray<btVector3> uniqueEdgesA;
                                        btAlignedObjectArray<btGpuFace> facesA;
                                        btAlignedObjectArray<int> indicesA;
                                        
                                        btAlignedObjectArray<btVector3> verticesA;
                                        
                                        
                                        
                                        
                                        btCollidable colA;
                                        colA.m_shapeIndex = 0;
                                        colA.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
                                        hostCollidablesA.push_back(colA);
                                        
                                        ConvexPolyhedronCL convexPolyhedronA;

                                        //add 3 vertices of the triangle
                                        convexPolyhedronA.m_numVertices = 3;
                                        convexPolyhedronA.m_vertexOffset = 0;
                                        for (int i=0;i<3;i++)
                                        {
                                            btVector3 vert(obj->vertexList[face->vertex_index[0]]->e[0],
                                                           obj->vertexList[face->vertex_index[1]]->e[1],
                                                           obj->vertexList[face->vertex_index[2]]->e[2]);
                                            verticesA.push_back(vert);
                                        }
                                        
                                        //a triangle has 3 unique edges
                                        convexPolyhedronA.m_numUniqueEdges = 3;
                                        convexPolyhedronA.m_uniqueEdgesOffset = 0;
                                        uniqueEdgesA.push_back(verticesA[1]-verticesA[0]);
                                        uniqueEdgesA.push_back(verticesA[2]-verticesA[1]);
                                        uniqueEdgesA.push_back(verticesA[0]-verticesA[2]);

                                        convexPolyhedronA.m_faceOffset = 0;
                                        
                                        btVector3 normal = ((verticesA[1]-verticesA[0]).cross(verticesA[2]-verticesA[0])).normalize();
                                        
                                        //front size of triangle
                                        {
                                            btGpuFace gpuFace;
                                            gpuFace.m_indexOffset=indicesA.size();
                                            indicesA.push_back(0);
                                            indicesA.push_back(1);
                                            indicesA.push_back(2);
                                            btScalar c = -(normal.dot(verticesA[0]));
                                            gpuFace.m_plane.x = normal[0];
                                            gpuFace.m_plane.y = normal[1];
                                            gpuFace.m_plane.z = normal[2];
                                            gpuFace.m_plane.w = c;
                                            gpuFace.m_numIndices=3;
                                            facesA.push_back(gpuFace);
                                        }

                                        //back size of triangle
                                        {
                                            btGpuFace gpuFace;
                                            gpuFace.m_indexOffset=indicesA.size();
                                            indicesA.push_back(2);
                                            indicesA.push_back(1);
                                            indicesA.push_back(0);
                                            btScalar c = (normal.dot(verticesA[0]));
                                            gpuFace.m_plane.x = -normal[0];
                                            gpuFace.m_plane.y = -normal[1];
                                            gpuFace.m_plane.z = -normal[2];
                                            gpuFace.m_plane.w = c;
                                            gpuFace.m_numIndices=3;
                                            facesA.push_back(gpuFace);
                                        }

                                        bool addEdgePlanes = true;
                                        if (addEdgePlanes)
                                        {
                                            int numVertices=3;
                                            int prevVertex = numVertices-1;
                                            for (int i=0;i<numVertices-1;i++)
                                            {
                                                btGpuFace gpuFace;
                                                
                                                btVector3 v0 = verticesA[prevVertex];
                                                btVector3 v1 = verticesA[i];
                                                
                                                btVector3 edgeNormal = (normal.cross(v1-v0)).normalize();
                                                btScalar c = -edgeNormal.dot(v0);

                                                gpuFace.m_numIndices = 2;
                                                gpuFace.m_indexOffset=indicesA.size();
                                                indicesA.push_back(prevVertex);
                                                indicesA.push_back(i);
                                                
                                                gpuFace.m_plane.x = edgeNormal[0];
                                                gpuFace.m_plane.y = edgeNormal[1];
                                                gpuFace.m_plane.z = edgeNormal[2];
                                                gpuFace.m_plane.w = c;
                                                facesA.push_back(gpuFace);
                                                prevVertex = i;
                                            }
                                        }
                                        convexPolyhedronA.m_numFaces = facesA.size();
                                        
                                        hostConvexDataA.push_back(convexPolyhedronA);
                                        
                                        
                                        m_internalData->m_gpuSatCollision->computeConvexConvexContactsGPUSATSingle(
                                                                                                                   bodyIndexA, bodyIndexB,
                                                                                                                   collidableIndexA, collidableIndexB,
                                                                                                                   m_internalData->m_bodyBufferCPU,
                                                                                                                   0,
                                                                                                                   m_internalData->m_pBufContactOutGPU,
                                                                                                                   nContactOut,cfgNP,
                                                                                                                   hostConvexDataA,
                                                                                                                   *m_internalData->m_convexPolyhedraGPU,
                                                                                                                   verticesA,
                                                                                                                   uniqueEdgesA,
                                                                                                                   facesA,
                                                                                                                   indicesA,
                                                                                                                   *m_internalData->m_convexVerticesGPU,
                                                                                                                   *m_internalData->m_uniqueEdgesGPU,
                                                                                                                   *m_internalData->m_convexFacesGPU,
                                                                                                                   *m_internalData->m_convexIndicesGPU,
                                                                                                                   hostCollidablesA,
                                                                                                                   *m_internalData->m_collidablesGPU);
                                        
                                        
                                        
                                        //indicesPtr->push_back(face->vertex_index[0]);
                                        //indicesPtr->push_back(face->vertex_index[1]);
                                        //indicesPtr->push_back(face->vertex_index[2]);
                                    }
                                    
                                }
                                
                                
                            }
                            
                            if (bodyB.m_shapeType == CollisionShape::SHAPE_CONCAVE_TRIMESH)
                            {
                                printf("concave y!\n");
                            }
                        }
                    }
                    
                    //if shapeType == concave for both
                    //find all overlapping triangles
                    //run convex-triangle for each pair
                    
                    //&broadphasePairsGPU, numBroadphasePairs,
                    //m_internalData->m_bodyBufferGPU,
                    
                    
                    
                }
            }
            else
            {
                if (useCulling)
                {
                    m_internalData->m_narrowPhase->execute(
                                                           m_internalData->m_convexPairsOutGPU,
                                                           numPairsOut,
                                                           m_internalData->m_bodyBufferGPU,
                                                           m_internalData->m_ShapeBuffer,
                                                           m_internalData->m_pBufContactOutGPU,
                                                           nContactOut, cfgNP);
                } else
                {
                    m_internalData->m_narrowPhase->execute(
                                                           &broadphasePairsGPU, numBroadphasePairs,
                                                           m_internalData->m_bodyBufferGPU,
                                                           m_internalData->m_ShapeBuffer,
                                                           m_internalData->m_pBufContactOutGPU,
                                                           nContactOut, cfgNP);
                }
            }
        }
        
        clFinish(m_queue);
    }
    
    
    m_internalData->m_pBufContactOutGPU->resize(nContactOut);
    
    if (!nContactOut)
        return;
    
    
    bool useSolver = true;//true;//false;
    
    if (useSolver)
    {
        float dt=1./60.;
        SolverBase::ConstraintCfg csCfg( dt );
        csCfg.m_enableParallelSolve = true;
        csCfg.m_averageExtent = 0.2f;//@TODO m_averageObjExtent;
        csCfg.m_staticIdx = m_static0Index;//m_planeBodyIndex;
        
        btOpenCLArray<Contact4>* contactsIn = m_internalData->m_pBufContactOutGPU;
        const btOpenCLArray<RigidBodyBase::Body>* bodyBuf = m_internalData->m_bodyBufferGPU;
        void* additionalData = m_internalData->m_frictionCGPU;
        const btOpenCLArray<RigidBodyBase::Inertia>* shapeBuf = m_internalData->m_inertiaBufferGPU;
        SolverData contactCOut = m_internalData->m_contactCGPU;
        int nContacts = nContactOut;
        
        bool useCPU=false;
        
        {
            
            if( m_internalData->m_solverGPU->m_contactBuffer)
            {
                m_internalData->m_solverGPU->m_contactBuffer->resize(nContacts);
            }
            
            if( m_internalData->m_solverGPU->m_contactBuffer == 0 )
            {
                m_internalData->m_solverGPU->m_contactBuffer = new btOpenCLArray<Contact4>(m_context,m_queue, nContacts );
                m_internalData->m_solverGPU->m_contactBuffer->resize(nContacts);
            }
            clFinish(m_queue);
            
            
            
            {
                BT_PROFILE("batching");
                //@todo: just reserve it, without copy of original contact (unless we use warmstarting)
                
                
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
                            
                            btAssert( sortSize%64 == 0 );
                            CB cdata;
                            cdata.m_nContacts = nContacts;
                            cdata.m_staticIdx = csCfg.m_staticIdx;
                            cdata.m_scale = 1.f/(BT_SOLVER_N_OBJ_PER_SPLIT*csCfg.m_averageExtent);
                            cdata.m_nSplit = BT_SOLVER_N_SPLIT;
                            
                            m_internalData->m_solverGPU->m_sortDataBuffer->resize(nContacts);
                            
                            
                            btBufferInfoCL bInfo[] = { btBufferInfoCL( contactNative->getBufferCL() ), btBufferInfoCL( bodyBuf->getBufferCL()), btBufferInfoCL( m_internalData->m_solverGPU->m_sortDataBuffer->getBufferCL()) };
                            btLauncherCL launcher(m_queue, m_internalData->m_solverGPU->m_setSortDataKernel );
                            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
                            launcher.setConst( cdata.m_nContacts );
                            launcher.setConst( cdata.m_scale );
                            launcher.setConst(cdata.m_nSplit);
                            
                            
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
                } else
                {
                    BT_PROFILE("cpu batchContacts");
                    btAlignedObjectArray<Contact4> cpuContacts;
                    btOpenCLArray<Contact4>* contactsIn = m_internalData->m_pBufContactOutGPU;
                    contactsIn->copyToHost(cpuContacts);
                    
                    btOpenCLArray<u32>* countsNative = m_internalData->m_solverGPU->m_numConstraints;
                    btOpenCLArray<u32>* offsetsNative = m_internalData->m_solverGPU->m_offsets;
                    
                    btAlignedObjectArray<u32> nNativeHost;
                    btAlignedObjectArray<u32> offsetsNativeHost;
                    
                    {
                        BT_PROFILE("countsNative/offsetsNative copyToHost");
                        countsNative->copyToHost(nNativeHost);
                        offsetsNative->copyToHost(offsetsNativeHost);
                    }
                    
                    
                    int numNonzeroGrid=0;
                    int maxNumBatches = 0;
                    {
                        BT_PROFILE("batch grid");
                        for(int i=0; i<BT_SOLVER_N_SPLIT*BT_SOLVER_N_SPLIT; i++)
                        {
                            int n = (nNativeHost)[i];
                            int offset = (offsetsNativeHost)[i];
                            
                            if( n )
                            {
                                numNonzeroGrid++;
                                //printf("cpu batch\n");
                                
                                int simdWidth = -1;
                                int numBatches = sortConstraintByBatch( &cpuContacts[0]+offset, n, simdWidth,m_internalData );	//	on GPU
                                maxNumBatches = btMax(numBatches,maxNumBatches);
                                
                                clFinish(m_queue);
                                
                            }
                        }
                    }
                    {
                        BT_PROFILE("m_contactBuffer->copyFromHost");
                        m_internalData->m_solverGPU->m_contactBuffer->copyFromHost(cpuContacts);
                    }
                    //	printf("maxNumBatches = %d\n", maxNumBatches);
                    
                }
                
                
                
                
                if (1)
                {
                    //BT_PROFILE("gpu convertToConstraints");
                    m_internalData->m_solverGPU->convertToConstraints( bodyBuf, shapeBuf, m_internalData->m_solverGPU->m_contactBuffer /*contactNative*/, contactCOut, additionalData, nContacts, csCfg );
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
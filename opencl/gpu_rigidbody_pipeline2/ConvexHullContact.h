
#ifndef _CONVEX_HULL_CONTACT_H
#define _CONVEX_HULL_CONTACT_H

#include "../broadphase_benchmark/btOpenCLArray.h"
#include "../../dynamics/basic_demo/Stubs/AdlRigidBody.h"
#include "../../dynamics/basic_demo/Stubs/ChNarrowphase.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "../gpu_rigidbody_pipeline/btConvexUtility.h"
#include "../gpu_rigidbody_pipeline2/ConvexPolyhedronCL.h"
#include "../broadphase_benchmark/btOpenCLArray.h"

struct ContactResult
{
	virtual ~ContactResult(){}	
	virtual void addContactPoint(const btVector3& normalOnBInWorld,const btVector3& pointInWorld,btScalar depth)=0;
};


struct GpuSatCollision
{
	cl_context				m_context;
	cl_device_id			m_device;
	cl_command_queue		m_queue;
	cl_kernel				m_findSeparatingAxisKernel;

	btAlignedObjectArray<Contact4>	m_hostContactOut;
	btAlignedObjectArray<int2>		m_hostPairs;

	GpuSatCollision(cl_context ctx,cl_device_id device, cl_command_queue  q );
	virtual ~GpuSatCollision();
	

	void computeConvexConvexContactsHost( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const ChNarrowphase::Config& cfg , 
			const btAlignedObjectArray<ConvexPolyhedronCL>* hostConvexData,
			const btAlignedObjectArray<btVector3>& vertices,
			const btAlignedObjectArray<btVector3>& uniqueEdges,
			const btAlignedObjectArray<btGpuFace>& faces,
			const btAlignedObjectArray<int>& indices);
};

#endif //_CONVEX_HULL_CONTACT_H
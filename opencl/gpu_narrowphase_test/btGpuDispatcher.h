#ifndef BT_GPU_DISPATCHER_H
#define BT_GPU_DISPATCHER_H

#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "../basic_initialize/btOpenCLInclude.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "../gpu_rigidbody_pipeline2/ConvexPolyhedronCL.h"
#include "../gpu_rigidbody_pipeline/btCollidable.h"

struct GpuSatCollision;

class btGpuDispatcher : public btCollisionDispatcher
{
protected:

	cl_context			m_ctx;
	cl_device_id		m_device;
	cl_command_queue	m_queue;

	struct GpuSatCollision* m_satCollision;

	btAlignedObjectArray<btVector3>				m_vertices;
	btAlignedObjectArray<btVector3>				m_uniqueEdges;
	btAlignedObjectArray<btGpuFace>				m_faces;
	btAlignedObjectArray<int>					m_indices;
	btAlignedObjectArray<ConvexPolyhedronCL>	m_hostConvexData;

	btAlignedObjectArray<btCollidable>			m_hostCollidables;

	public:
	btGpuDispatcher(btCollisionConfiguration* collisionConfiguration,cl_context ctx,cl_device_id device, cl_command_queue  q);

	virtual ~btGpuDispatcher();

	virtual void	dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,const btDispatcherInfo& dispatchInfo,btDispatcher* dispatcher) ;

};

#endif //BT_GPU_DISPATCHER_H

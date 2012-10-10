#ifndef BT_GPU_DISPATCHER_H
#define BT_GPU_DISPATCHER_H

#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "../basic_initialize/btOpenCLInclude.h"

struct GpuSatCollision;

class btGpuDispatcher : public btCollisionDispatcher
{
	struct GpuSatCollision* m_satCollision;

	public:
	btGpuDispatcher(btCollisionConfiguration* collisionConfiguration,cl_context ctx,cl_device_id device, cl_command_queue  q);

	virtual ~btGpuDispatcher();

	virtual void	dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,const btDispatcherInfo& dispatchInfo,btDispatcher* dispatcher) ;

};

#endif //BT_GPU_DISPATCHER_H

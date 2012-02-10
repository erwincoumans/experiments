#ifndef GPU_NARROWPHASE_SOLVER_H
#define GPU_NARROWPHASE_SOLVER_H



//#define MAX_CONVEX_BODIES_CL 8*1024
#define MAX_CONVEX_BODIES_CL 128*1024
#define MAX_PAIRS_PER_BODY_CL 16
#define MAX_CONVEX_SHAPES_CL 8192
#define MAX_BROADPHASE_COLLISION_CL (MAX_CONVEX_BODIES_CL*MAX_PAIRS_PER_BODY_CL)

/*
#define MAX_CONVEX_BODIES_CL 1024
#define MAX_PAIRS_PER_BODY_CL 32
#define MAX_CONVEX_SHAPES_CL 8192
#define MAX_BROADPHASE_COLLISION_CL (MAX_CONVEX_BODIES_CL*MAX_PAIRS_PER_BODY_CL)
*/

namespace adl
{
	struct DeviceCL;
};


struct	CustomDispatchData;

#include "../basic_initialize/btOpenCLInclude.h"


class btGpuNarrowphaseAndSolver
{
protected:

	CustomDispatchData*	m_internalData;
	int m_acceleratedCompanionShapeIndex;
	int m_planeBodyIndex;

public:
	btGpuNarrowphaseAndSolver(adl::DeviceCL* deviceCL);

	virtual ~btGpuNarrowphaseAndSolver(void);

	int registerShape(class ConvexHeightField* convexShape);
	int registerRigidBody(int shapeIndex, float mass, const float* position, const float* orientation, bool writeToGpu = true);
	void	writeAllBodiesToGpu();
	
	//btBroadphasePair* GetPair(btBroadphasePairArray& pairArray, int idxBodyA, int idxBodyB);

	virtual void computeContactsAndSolver(cl_mem broadphasePairs, int numBroadphasePairs);

	cl_mem	getBodiesGpu();

	cl_mem	getBodyInertiasGpu();

};

#endif //GPU_NARROWPHASE_SOLVER_H

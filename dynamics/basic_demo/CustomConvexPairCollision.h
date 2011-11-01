#ifndef CUSTOM_CONVEX_CONVEX_PAIR_COLLISION_H
#define CUSTOM_CONVEX_CONVEX_PAIR_COLLISION_H


#include "BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h"

class CustomConvexConvexPairCollision : public btConvexConvexAlgorithm
{
	public:

	CustomConvexConvexPairCollision(btPersistentManifold* mf,const btCollisionAlgorithmConstructionInfo& ci,btCollisionObject* body0,btCollisionObject* body1, btSimplexSolverInterface* simplexSolver, btConvexPenetrationDepthSolver* pdSolver, int numPerturbationIterations, int minimumPointsPerturbationThreshold);
	virtual ~CustomConvexConvexPairCollision();

	virtual void processCollision (btCollisionObject* body0,btCollisionObject* body1,const btDispatcherInfo& dispatchInfo,btManifoldResult* resultOut);


	struct CreateFunc :public 	btConvexConvexAlgorithm::CreateFunc
	{

		CreateFunc(btSimplexSolverInterface*			simplexSolver, btConvexPenetrationDepthSolver* pdSolver);
		
		virtual ~CreateFunc();

		virtual	btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci, btCollisionObject* body0,btCollisionObject* body1)
		{
			void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(CustomConvexConvexPairCollision));
			return new(mem) CustomConvexConvexPairCollision(ci.m_manifold,ci,body0,body1,m_simplexSolver,m_pdSolver,m_numPerturbationIterations,m_minimumPointsPerturbationThreshold);
		}
	};
	

};


#endif //CUSTOM_CONVEX_CONVEX_PAIR_COLLISION_H

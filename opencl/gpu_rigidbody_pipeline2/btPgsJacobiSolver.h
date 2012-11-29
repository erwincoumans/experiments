#ifndef BT_PGS_JACOBI_SOLVER
#define BT_PGS_JACOBI_SOLVER

namespace RigidBodyBase
{

	struct Body;
	struct Inertia;
};
struct Contact4;

struct btContactPoint;


class btDispatcher;

#include "BulletDynamics/ConstraintSolver/btTypedConstraint.h"
#include "BulletDynamics/ConstraintSolver/btContactSolverInfo.h"
#include "BulletDynamics/ConstraintSolver/btSolverBody.h"
#include "BulletDynamics/ConstraintSolver/btSolverConstraint.h"
#include "BulletDynamics/ConstraintSolver/btConstraintSolver.h"


class btPgsJacobiSolver
{

protected:
	btAlignedObjectArray<btSolverBody>      m_tmpSolverBodyPool;
	btConstraintArray			m_tmpSolverContactConstraintPool;
	btConstraintArray			m_tmpSolverNonContactConstraintPool;
	btConstraintArray			m_tmpSolverContactFrictionConstraintPool;
	btConstraintArray			m_tmpSolverContactRollingFrictionConstraintPool;

	btAlignedObjectArray<int>	m_orderTmpConstraintPool;
	btAlignedObjectArray<int>	m_orderNonContactConstraintPool;
	btAlignedObjectArray<int>	m_orderFrictionConstraintPool;
	btAlignedObjectArray<btTypedConstraint::btConstraintInfo1> m_tmpConstraintSizesPool;
	int							m_maxOverrideNumSolverIterations;
	btScalar	getContactProcessingThreshold(Contact4* contact)
	{
		return 0.0f;
	}
	void setupFrictionConstraint(	RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias, btSolverConstraint& solverConstraint, const btVector3& normalAxis,int solverBodyIdA,int  solverBodyIdB,
									btContactPoint& cp,const btVector3& rel_pos1,const btVector3& rel_pos2,
									RigidBodyBase::Body* colObj0,RigidBodyBase::Body* colObj1, btScalar relaxation, 
									btScalar desiredVelocity=0., btScalar cfmSlip=0.);

	void setupRollingFrictionConstraint(RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias,	btSolverConstraint& solverConstraint, const btVector3& normalAxis,int solverBodyIdA,int  solverBodyIdB,
									btContactPoint& cp,const btVector3& rel_pos1,const btVector3& rel_pos2,
									RigidBodyBase::Body* colObj0,RigidBodyBase::Body* colObj1, btScalar relaxation, 
									btScalar desiredVelocity=0., btScalar cfmSlip=0.);

	btSolverConstraint&	addFrictionConstraint(RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias,const btVector3& normalAxis,int solverBodyIdA,int solverBodyIdB,int frictionIndex,btContactPoint& cp,const btVector3& rel_pos1,const btVector3& rel_pos2,RigidBodyBase::Body* colObj0,RigidBodyBase::Body* colObj1, btScalar relaxation, btScalar desiredVelocity=0., btScalar cfmSlip=0.);
	btSolverConstraint&	addRollingFrictionConstraint(RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias,const btVector3& normalAxis,int solverBodyIdA,int solverBodyIdB,int frictionIndex,btContactPoint& cp,const btVector3& rel_pos1,const btVector3& rel_pos2,RigidBodyBase::Body* colObj0,RigidBodyBase::Body* colObj1, btScalar relaxation, btScalar desiredVelocity=0, btScalar cfmSlip=0.f);


	void setupContactConstraint(RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias,
								btSolverConstraint& solverConstraint, int solverBodyIdA, int solverBodyIdB, btContactPoint& cp, 
								const btContactSolverInfo& infoGlobal, btVector3& vel, btScalar& rel_vel, btScalar& relaxation, 
								btVector3& rel_pos1, btVector3& rel_pos2);

	void setFrictionConstraintImpulse( RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias,btSolverConstraint& solverConstraint, int solverBodyIdA,int solverBodyIdB, 
										 btContactPoint& cp, const btContactSolverInfo& infoGlobal);

	///m_btSeed2 is used for re-arranging the constraint rows. improves convergence/quality of friction
	unsigned long	m_btSeed2;

	
	btScalar restitutionCurve(btScalar rel_vel, btScalar restitution);

	void	convertContact(RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias,Contact4* manifold,const btContactSolverInfo& infoGlobal);


	void	resolveSplitPenetrationSIMD(
     btSolverBody& bodyA,btSolverBody& bodyB,
        const btSolverConstraint& contactConstraint);

	void	resolveSplitPenetrationImpulseCacheFriendly(
       btSolverBody& bodyA,btSolverBody& bodyB,
        const btSolverConstraint& contactConstraint);

	//internal method
	int		getOrInitSolverBody(int bodyIndex, RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias);
	void	initSolverBody(int bodyIndex, btSolverBody* solverBody, RigidBodyBase::Body* collisionObject);

	void	resolveSingleConstraintRowGeneric(btSolverBody& bodyA,btSolverBody& bodyB,const btSolverConstraint& contactConstraint);

	void	resolveSingleConstraintRowGenericSIMD(btSolverBody& bodyA,btSolverBody& bodyB,const btSolverConstraint& contactConstraint);
	
	void	resolveSingleConstraintRowLowerLimit(btSolverBody& bodyA,btSolverBody& bodyB,const btSolverConstraint& contactConstraint);
	
	void	resolveSingleConstraintRowLowerLimitSIMD(btSolverBody& bodyA,btSolverBody& bodyB,const btSolverConstraint& contactConstraint);
		
protected:

	virtual btScalar solveGroupCacheFriendlySetup(RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias,int numBodies,Contact4* manifoldPtr, int numManifolds,btTypedConstraint** constraints,int numConstraints,const btContactSolverInfo& infoGlobal);


	virtual btScalar solveGroupCacheFriendlyIterations(btTypedConstraint** constraints,int numConstraints,const btContactSolverInfo& infoGlobal);
	virtual void solveGroupCacheFriendlySplitImpulseIterations(btTypedConstraint** constraints,int numConstraints,const btContactSolverInfo& infoGlobal);
	btScalar solveSingleIteration(int iteration, btTypedConstraint** constraints,int numConstraints,const btContactSolverInfo& infoGlobal);


	virtual btScalar solveGroupCacheFriendlyFinish(RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias,int numBodies,const btContactSolverInfo& infoGlobal);


public:

	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btPgsJacobiSolver();
	virtual ~btPgsJacobiSolver();

	void	solveContacts(int numBodies, RigidBodyBase::Body* bodies, RigidBodyBase::Inertia* inertias, int numContacts, Contact4* contacts);
	
	btScalar solveGroup(RigidBodyBase::Body* bodies,RigidBodyBase::Inertia* inertias,int numBodies,Contact4* manifoldPtr, int numManifolds,btTypedConstraint** constraints,int numConstraints,const btContactSolverInfo& infoGlobal);

	///clear internal cached data and reset random seed
	virtual	void	reset();
	
	unsigned long btRand2();

	int btRandInt2 (int n);

	void	setRandSeed(unsigned long seed)
	{
		m_btSeed2 = seed;
	}
	unsigned long	getRandSeed() const
	{
		return m_btSeed2;
	}




};

#endif //BT_PGS_JACOBI_SOLVER


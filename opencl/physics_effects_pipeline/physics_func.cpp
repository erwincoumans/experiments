/*
Physics Effects Copyright(C) 2010 Sony Computer Entertainment Inc.
All rights reserved.

Physics Effects is open software; you can redistribute it and/or
modify it under the terms of the BSD License.

Physics Effects is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the BSD License for more details.

A copy of the BSD License is distributed with
Physics Effects under the filename: physics_effects_license.txt
*/

#include "physics_func.h"
#include "LinearMath/btQuickprof.h"
#include "btPgsSolver.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "btFakeRigidBody.h"
btAlignedObjectArray<btRigidBody> rbs;
btAlignedObjectArray<btPersistentManifold> manifolds;
bool peSolverEnabled=false;


//#include "sample_api_physics_effects/common/perf_func.h"

///////////////////////////////////////////////////////////////////////////////
// Simulation Data

#define NUM_RIGIDBODIES 5000
#define NUM_JOINTS    5000
#define NUM_CONTACTS  16000

const float timeStep = 0.016f;
const float separateBias = 0.01f;
int iteration = 10;

//J ワールドサイズ
//E World size
PfxVector3 worldCenter(0.0f);
PfxVector3 worldExtent(500.0f);

//J 剛体
//E Rigid body
PfxRigidState states[NUM_RIGIDBODIES];
PfxRigidBody  bodies[NUM_RIGIDBODIES];
PfxCollidable collidables[NUM_RIGIDBODIES];
PfxSolverBody solverBodies[NUM_RIGIDBODIES];
int numRigidBodies = 0;

//J 地形を表現するためのラージメッシュ
//E Large mesh for representing a landscape
#include "landscape.h"
PfxLargeTriMesh gLargeMesh;

//J 凸メッシュ
//E Convex Mesh
#include "barrel.h"
PfxConvexMesh gConvex;

//J プロキシ
//E Proxies
PfxBroadphaseProxy proxies[NUM_RIGIDBODIES];

//J ジョイント
//E Joint
PfxConstraintPair jointPairs[NUM_JOINTS];
PfxJoint joints[NUM_JOINTS];
int numJoints = 0;

//J ペア
//E Pairs
unsigned int pairSwap;
unsigned int numPairs[2];
PfxBroadphasePair pairsBuff[2][NUM_CONTACTS];

//J コンタクト
//E Contacts
PfxContactManifold contacts[NUM_CONTACTS];
int numContacts;

PfxUInt32 contactIdPool[NUM_CONTACTS];
int numContactIdPool;

//J 一時バッファ
//E Temporary buffers
#define POOL_BYTES (16*1024*1024)
unsigned char SCE_PFX_ALIGNED(128) poolBuff[POOL_BYTES];

//J 一時バッファ用スタックアロケータ
//E Stack allocator for temporary buffers
PfxHeapManager pool(poolBuff,POOL_BYTES);

///////////////////////////////////////////////////////////////////////////////
// Simulation Function

int frame = 0;
int curNumPairs = 0;
int curTotalContacts = 0;

void broadphase()
{
	pairSwap = 1-pairSwap;

	unsigned int &numPreviousPairs = numPairs[1-pairSwap];
	unsigned int &numCurrentPairs = numPairs[pairSwap];
	PfxBroadphasePair *previousPairs = pairsBuff[1-pairSwap];
	PfxBroadphasePair *currentPairs = pairsBuff[pairSwap];

	//J 剛体が最も分散している軸を見つける
	//E Find the axis along which all rigid bodies are most widely positioned
	int axis = 0;
	{
		PfxVector3 s(0.0f),s2(0.0f);
		for(int i=0;i<numRigidBodies;i++) {
			PfxVector3 c = states[i].getPosition();
			s += c;
			s2 += mulPerElem(c,c);
		}
		PfxVector3 v = s2 - mulPerElem(s,s) / (float)numRigidBodies;
		if(v[1] > v[0]) axis = 1;
		if(v[2] > v[axis]) axis = 2;
	}

	//J ブロードフェーズプロキシの更新
	//E Create broadpahse proxies
	{
		for(int i=0;i<numRigidBodies;i++) {
			pfxUpdateBroadphaseProxy(proxies[i],states[i],collidables[i],worldCenter,worldExtent,axis);
		}

		int workBytes = sizeof(PfxBroadphaseProxy) * numRigidBodies;
		void *workBuff = pool.allocate(workBytes);
				
		pfxParallelSort(proxies,numRigidBodies,workBuff,workBytes);

		pool.deallocate(workBuff);
	}

	//J 交差ペア探索
	//E Find overlapped pairs
	{
		PfxFindPairsParam findPairsParam;
		findPairsParam.pairBytes = pfxGetPairBytesOfFindPairs(NUM_CONTACTS);
		findPairsParam.pairBuff = pool.allocate(findPairsParam.pairBytes);
		findPairsParam.workBytes = pfxGetWorkBytesOfFindPairs(NUM_CONTACTS);
		findPairsParam.workBuff = pool.allocate(findPairsParam.workBytes);
		findPairsParam.proxies = proxies;
		findPairsParam.numProxies = numRigidBodies;
		findPairsParam.maxPairs = NUM_CONTACTS;
		findPairsParam.axis = axis;

		PfxFindPairsResult findPairsResult;

		int ret = pfxFindPairs(findPairsParam,findPairsResult);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxFindPairs failed %d\n",ret);
		
		pool.deallocate(findPairsParam.workBuff);

		curNumPairs = findPairsResult.numPairs;

		//J 交差ペア合成
		//E Decompose overlapped pairs into 3 arrays
		PfxDecomposePairsParam decomposePairsParam;
		decomposePairsParam.pairBytes = pfxGetPairBytesOfDecomposePairs(numPreviousPairs,findPairsResult.numPairs);
		decomposePairsParam.pairBuff = pool.allocate(decomposePairsParam.pairBytes);
		decomposePairsParam.workBytes = pfxGetWorkBytesOfDecomposePairs(numPreviousPairs,findPairsResult.numPairs);
		decomposePairsParam.workBuff = pool.allocate(decomposePairsParam.workBytes);
		decomposePairsParam.previousPairs = previousPairs;
		decomposePairsParam.numPreviousPairs = numPreviousPairs;
		decomposePairsParam.currentPairs = findPairsResult.pairs; // Set pairs from pfxFindPairs()
		decomposePairsParam.numCurrentPairs = findPairsResult.numPairs; // Set the number of pairs from pfxFindPairs()

		PfxDecomposePairsResult decomposePairsResult;

		ret = pfxDecomposePairs(decomposePairsParam,decomposePairsResult);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxDecomposePairs failed %d\n",ret);

		pool.deallocate(decomposePairsParam.workBuff);

		PfxBroadphasePair *outNewPairs = decomposePairsResult.outNewPairs;
		PfxBroadphasePair *outKeepPairs = decomposePairsResult.outKeepPairs;
		PfxBroadphasePair *outRemovePairs = decomposePairsResult.outRemovePairs;
		PfxUInt32 numOutNewPairs = decomposePairsResult.numOutNewPairs;
		PfxUInt32 numOutKeepPairs = decomposePairsResult.numOutKeepPairs;
		PfxUInt32 numOutRemovePairs = decomposePairsResult.numOutRemovePairs;

		//J 廃棄ペアのコンタクトをプールに戻す
		//E Put removed contacts into the contact pool
		for(PfxUInt32 i=0;i<numOutRemovePairs;i++) {
			contactIdPool[numContactIdPool++] = pfxGetContactId(outRemovePairs[i]);
		}

		//J 新規ペアのコンタクトのリンクと初期化
		//E Add new contacts and initialize
		for(PfxUInt32 i=0;i<numOutNewPairs;i++) {
			int cId = 0;
			if(numContactIdPool > 0) {
				cId = contactIdPool[--numContactIdPool];
			}
			else {
				cId = numContacts++;
			}
			if(cId >= NUM_CONTACTS) {
				cId = 0;
			}
			SCE_PFX_ASSERT(cId < NUM_CONTACTS);
			pfxSetContactId(outNewPairs[i],cId);
			PfxContactManifold &contact = contacts[cId];
			contact.reset(pfxGetObjectIdA(outNewPairs[i]),pfxGetObjectIdB(outNewPairs[i]));
		}

		//J 新規ペアと維持ペアを合成
		//E Merge 'new' and 'keep' pairs
		numCurrentPairs = 0;
		for(PfxUInt32 i=0;i<numOutKeepPairs;i++) {
			currentPairs[numCurrentPairs++] = outKeepPairs[i];
		}
		for(PfxUInt32 i=0;i<numOutNewPairs;i++) {
			currentPairs[numCurrentPairs++] = outNewPairs[i];
		}

		bool verboseStats = true;
		if (verboseStats)
		{
			printf("===============================================\n");
			printf("num bodies/states = %d\n", physics_get_num_rigidbodies());
			for (int i=0;i<physics_get_num_rigidbodies();i++)
			{
				PfxVector3 pos = physics_get_state(i).getPosition();
				printf("body %d has position %f,%f,%f\n",i,pos.getX(),pos.getY(),pos.getZ());
			}
			printf("numCurrentPairs (total) = %d\n", numCurrentPairs);

			for (int i=0;i<numCurrentPairs;i++)
			{
				int idA = pfxGetObjectIdA(currentPairs[i]);
				int idB = pfxGetObjectIdB(currentPairs[i]);
				printf("pfx pair[%d] idA = %d, idB = %d\n", i, idA,idB);
				int cId = pfxGetContactId(currentPairs[i]);
				printf("contact duration = %d\n", contacts[cId].getDuration());

			
				if (1)
				{
					printf("num contacts = %d\n", contacts[cId].getNumContacts());
					for (int c=0;c<contacts[cId].getNumContacts();c++)
					{
						const PfxContactPoint& cp = contacts[cId].getContactPoint(c);
						printf("localPosA = %f,%f,%f. ", cp.m_localPointA[0],cp.m_localPointA[1],cp.m_localPointA[2]);
						printf("localPosB = %f,%f,%f. ", cp.m_localPointB[0],cp.m_localPointB[1],cp.m_localPointB[2]);
						for (int r=0;r<3;r++)
						{
							printf("row %d accumImpulse = %f. ", r, cp.m_constraintRow[r].m_accumImpulse);
							printf("row %d normal = %f,%f,%f. ", r, cp.m_constraintRow[r].m_normal[0],cp.m_constraintRow[r].m_normal[1],cp.m_constraintRow[r].m_normal[2]);
							printf("row %d distance %f and duration %d\n", r, cp.m_distance1,cp.m_duration);

						}
					}
				}

			}
		}

		
		//printf("numOutRemovePairs = %d\n", numOutRemovePairs);
		//printf("numOutNewPairs = %d\n",numOutNewPairs);


		pool.deallocate(decomposePairsParam.pairBuff);
		pool.deallocate(findPairsParam.pairBuff);
	}
	
	{
		int workBytes = sizeof(PfxBroadphasePair) * numCurrentPairs;
		void *workBuff = pool.allocate(workBytes);
		
		pfxParallelSort(currentPairs,numCurrentPairs,workBuff,workBytes);
		
		pool.deallocate(workBuff);
	}
}

void collision()
{
	unsigned int numCurrentPairs = numPairs[pairSwap];
	PfxBroadphasePair *currentPairs = pairsBuff[pairSwap];
	
	//J 衝突検出
	//E Detect collisions
	{
		PfxDetectCollisionParam param;
		param.contactPairs = currentPairs;
		param.numContactPairs = numCurrentPairs;
		param.offsetContactManifolds = contacts;
		param.offsetRigidStates = states;
		param.offsetCollidables = collidables;
		param.numRigidBodies = numRigidBodies;

		int ret = pfxDetectCollision(param);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxDetectCollision failed %d\n",ret);
	}

	curTotalContacts = 0;

	for(PfxUInt32 i=0;i<numCurrentPairs;i++) {
		PfxConstraintPair &pair = currentPairs[i];
	
		PfxUInt16 iA = pfxGetObjectIdA(pair);
		PfxUInt16 iB = pfxGetObjectIdB(pair);
		PfxUInt32 iConstraint = pfxGetConstraintId(pair);

		PfxContactManifold &contact = contacts[iConstraint];
		curTotalContacts += contact.getNumContacts();
	}


	//J リフレッシュ
	//E Refresh contacts
	{
		PfxRefreshContactsParam param;
		param.contactPairs = currentPairs;
		param.numContactPairs = numCurrentPairs;
		param.offsetContactManifolds = contacts;
		param.offsetRigidStates = states;
		param.numRigidBodies = numRigidBodies;

		int ret = pfxRefreshContacts(param);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxRefreshContacts failed %d\n",ret);
	}


	

}




PfxInt32 BulletSetupContactConstraints(PfxSetupContactConstraintsParam &param)
{
//	PfxInt32 ret = pfxCheckParamOfSetupContactConstraints(param);
	//if(ret != SCE_PFX_OK) return ret;
	
	SCE_PFX_PUSH_MARKER("pfxSetupContactConstraints");

	PfxConstraintPair *contactPairs = param.contactPairs;
	PfxUInt32 numContactPairs = param.numContactPairs;
	PfxContactManifold *offsetContactManifolds = param.offsetContactManifolds;
	PfxRigidState *offsetRigidStates = param.offsetRigidStates;
	PfxRigidBody *offsetRigidBodies = param.offsetRigidBodies;
	PfxSolverBody *offsetSolverBodies = param.offsetSolverBodies;
	manifolds.resize(0);

	for(PfxUInt32 i=0;i<numContactPairs;i++) {
		PfxConstraintPair &pair = contactPairs[i];

//		if(!sce::PhysicsEffects::pfxCheckSolver(pair)) {
	//		continue;
		//}

		PfxUInt16 iA = pfxGetObjectIdA(pair);
		PfxUInt16 iB = pfxGetObjectIdB(pair);
		PfxUInt32 iConstraint = pfxGetConstraintId(pair);

		PfxContactManifold &contact = offsetContactManifolds[iConstraint];

		btPersistentManifold& manifold = manifolds.expand();
		memset(&manifold,0xff,sizeof(btPersistentManifold));

		manifold.m_body0 = &rbs[iA];
		manifold.m_body1 = &rbs[iB];
		manifold.m_cachedPoints = contact.getNumContacts();

		if (!contact.getNumContacts())
			continue;


		SCE_PFX_ALWAYS_ASSERT(iA==contact.getRigidBodyIdA());
		SCE_PFX_ALWAYS_ASSERT(iB==contact.getRigidBodyIdB());

		PfxRigidState &stateA = offsetRigidStates[iA];
		PfxRigidBody &bodyA = offsetRigidBodies[iA];
		PfxSolverBody &solverBodyA = offsetSolverBodies[iA];

		PfxRigidState &stateB = offsetRigidStates[iB];
		PfxRigidBody &bodyB = offsetRigidBodies[iB];
		PfxSolverBody &solverBodyB = offsetSolverBodies[iB];
	
		contact.setInternalFlag(0);
		
		PfxFloat restitution = 0.5f * (bodyA.getRestitution() + bodyB.getRestitution());
		if(contact.getDuration() > 1) restitution = 0.0f;
		
		PfxFloat friction = sqrtf(bodyA.getFriction() * bodyB.getFriction());
		
		manifold.m_cachedPoints = contact.getNumContacts();
	

		manifold.m_contactProcessingThreshold = 0.01f;//SCE_PFX_CONTACT_THRESHOLD_NORMAL;
		manifold.m_contactBreakingThreshold = 0.01f;

		for(int j=0;j<contact.getNumContacts();j++) {
			PfxContactPoint &cp = contact.getContactPoint(j);

			PfxVector3 ptA = pfxReadVector3(cp.m_localPointA);
			manifold.m_pointCache[j].m_localPointA.setValue(ptA.getX(),ptA.getY(),ptA.getZ());
			PfxVector3 ptB = pfxReadVector3(cp.m_localPointB);
			manifold.m_pointCache[j].m_localPointB.setValue(ptB.getX(),ptB.getY(),ptB.getZ());
			
			manifold.m_pointCache[j].m_normalWorldOnB.setValue(
						cp.m_constraintRow[0].m_normal[0],
						cp.m_constraintRow[0].m_normal[1],
						cp.m_constraintRow[0].m_normal[2]);
			manifold.m_pointCache[j].m_distance1 = cp.m_distance1;
			manifold.m_pointCache[j].m_combinedFriction = friction;
			manifold.m_pointCache[j].m_combinedRestitution = restitution;
			manifold.m_pointCache[j].m_appliedImpulse = cp.m_constraintRow[0].m_accumImpulse;
			manifold.m_pointCache[j].m_lateralFrictionDir1.setValue(
						cp.m_constraintRow[1].m_normal[0],
						cp.m_constraintRow[1].m_normal[1],
						cp.m_constraintRow[1].m_normal[2]);
			manifold.m_pointCache[j].m_appliedImpulseLateral1 = cp.m_constraintRow[1].m_accumImpulse;

			manifold.m_pointCache[j].m_lateralFrictionDir2.setValue(
						cp.m_constraintRow[2].m_normal[0],
						cp.m_constraintRow[2].m_normal[1],
						cp.m_constraintRow[2].m_normal[2]);
			manifold.m_pointCache[j].m_appliedImpulseLateral2 = cp.m_constraintRow[2].m_accumImpulse;
			manifold.m_pointCache[j].m_lateralFrictionInitialized = true;
			manifold.m_pointCache[j].m_lifeTime = cp.m_duration;

			btTransform trA = manifold.m_body0->getWorldTransform();
			btTransform trB = manifold.m_body1->getWorldTransform();

			manifold.m_pointCache[j].m_positionWorldOnA = trA( manifold.m_pointCache[j].m_localPointA );
			manifold.m_pointCache[j].m_positionWorldOnB = trB( manifold.m_pointCache[j].m_localPointB );




						//btVector3 m_localPointA;			
			//btVector3 m_localPointB;			
			//btVector3	m_positionWorldOnB;
			//m_positionWorldOnA is redundant information, see getPositionWorldOnA(), but for clarity
			//btVector3	m_positionWorldOnA;




			/*
			pfxSetupContactConstraint(
				cp.m_constraintRow[0],
				cp.m_constraintRow[1],
				cp.m_constraintRow[2],
				cp.m_distance,
				restitution,
				friction,
				pfxReadVector3(cp.m_constraintRow[0].m_normal),
				pfxReadVector3(cp.m_localPointA),
				pfxReadVector3(cp.m_localPointB),
				stateA,
				stateB,
				solverBodyA,
				solverBodyB,
				param.separateBias,
				param.timeStep
				);
				*/

		}

		contact.setCompositeFriction(friction);
	}

	SCE_PFX_POP_MARKER();

	return SCE_PFX_OK;
}




PfxInt32 BulletWriteWarmstartContactConstraints(PfxSetupContactConstraintsParam &param)
{
//	PfxInt32 ret = pfxCheckParamOfSetupContactConstraints(param);
	//if(ret != SCE_PFX_OK) return ret;
	
	SCE_PFX_PUSH_MARKER("pfxSetupContactConstraints");

	PfxConstraintPair *contactPairs = param.contactPairs;
	PfxUInt32 numContactPairs = param.numContactPairs;
	PfxContactManifold *offsetContactManifolds = param.offsetContactManifolds;
	PfxRigidState *offsetRigidStates = param.offsetRigidStates;
	PfxRigidBody *offsetRigidBodies = param.offsetRigidBodies;
	PfxSolverBody *offsetSolverBodies = param.offsetSolverBodies;

	for(PfxUInt32 i=0;i<numContactPairs;i++) 
	{
		PfxConstraintPair &pair = contactPairs[i];

		PfxUInt16 iA = pfxGetObjectIdA(pair);
		PfxUInt16 iB = pfxGetObjectIdB(pair);
		PfxUInt32 iConstraint = pfxGetConstraintId(pair);

		PfxContactManifold &contact = offsetContactManifolds[iConstraint];
		btPersistentManifold& manifold = manifolds[i];
		for (int c=0;c<manifold.m_cachedPoints;c++)
		{
			contact.getContactPoint(c).m_constraintRow[0].m_accumImpulse = manifold.m_pointCache[c].m_appliedImpulse;
			contact.getContactPoint(c).m_constraintRow[1].m_accumImpulse = manifold.m_pointCache[c].m_appliedImpulseLateral1;
			contact.getContactPoint(c).m_constraintRow[2].m_accumImpulse = manifold.m_pointCache[c].m_appliedImpulseLateral2;
		}


	}
	return 0;
}


void BulletConstraintSolver()
{
	btPgsSolver pgs;
	btContactSolverInfo info;
	rbs.resize(0);

	for (int i=0;i<numRigidBodies;i++)
	{
		btRigidBody& rb = rbs.expandNonInitializing();

		rb.m_companionId=-1;
		rb.m_angularFactor.setValue(1,1,1);
		rb.m_anisotropicFriction.setValue(1,1,1);
		
		rb.m_invMass = bodies[i].getMassInv();
		rb.m_linearFactor.setValue(1,1,1);
		btVector3 pos(states[i].getPosition().getX(),states[i].getPosition().getY(),states[i].getPosition().getZ());
		rb.m_worldTransform.setIdentity();
		btQuaternion orn(states[i].getOrientation().getX(),states[i].getOrientation().getY(),states[i].getOrientation().getZ(),states[i].getOrientation().getW());
		rb.m_worldTransform.setRotation(orn);
		rb.m_worldTransform.setOrigin(pos);
		PfxMatrix3 ori(states[i].getOrientation());
		rb.m_worldTransform.setRotation(orn);

		PfxMatrix3 inertiaInvWorld = ori * bodies[i].getInertiaInv() * transpose(ori);
		rb.m_invInertiaWorld.setIdentity();

		if (rb.m_invMass)
		{
			for (int row=0;row<3;row++)
			{
				for (int col=0;col<3;col++)
				{
					rb.m_invInertiaWorld[col][row] = inertiaInvWorld.getElem(col,row);
				}
			}
		} else
		{
			rb.m_invInertiaWorld = btMatrix3x3(0,0,0,0,0,0,0,0,0);
		}
		rb.m_linearVelocity.setValue(states[i].getLinearVelocity().getX(),states[i].getLinearVelocity().getY(),states[i].getLinearVelocity().getZ());
		rb.m_angularVelocity.setValue(states[i].getAngularVelocity().getX(),states[i].getAngularVelocity().getY(),states[i].getAngularVelocity().getZ());
//		printf("body added\n");
	}

	btAlignedObjectArray<btCollisionObject*> bodyPtrs;
	bodyPtrs.resize(rbs.size());
	for (int i=0;i<rbs.size();i++)
	{
		bodyPtrs[i] = &rbs[i];
	}


	unsigned int numCurrentPairs = numPairs[pairSwap];
	PfxBroadphasePair *currentPairs = pairsBuff[pairSwap];

	PfxSetupContactConstraintsParam param;
	param.contactPairs = currentPairs;
	param.numContactPairs = numCurrentPairs;
	param.offsetContactManifolds = contacts;
	param.offsetRigidStates = states;
	param.offsetRigidBodies = bodies;
	param.offsetSolverBodies = solverBodies;
	param.numRigidBodies = numRigidBodies;
	param.timeStep = timeStep;
	param.separateBias = separateBias;

	BulletSetupContactConstraints(param);

	btAlignedObjectArray<btPersistentManifold*> manifoldPtrs;
	manifoldPtrs.resize(manifolds.size());

	for (int i=0;i<manifolds.size();i++)
	{
		manifoldPtrs[i] = &manifolds[i];
	}

	if (bodyPtrs.size() && manifoldPtrs.size())
	{
		pgs.solveGroup(&bodyPtrs[0],bodyPtrs.size(),&manifoldPtrs[0],manifoldPtrs.size(),0,0,info,0,0,0);

		for (int i=0;i<numRigidBodies;i++)
		{
			btVector3 linvel = rbs[i].getLinearVelocity();
			btVector3 angvel = rbs[i].getAngularVelocity();
			states[i].setLinearVelocity(PfxVector3(linvel.getX(),linvel.getY(),linvel.getZ()));
			states[i].setAngularVelocity(PfxVector3(angvel.getX(),angvel.getY(),angvel.getZ()));
		}
	}

	BulletWriteWarmstartContactConstraints(param);



}
void constraintSolver()
{
	PfxPerfCounter pc;

	unsigned int numCurrentPairs = numPairs[pairSwap];
	PfxBroadphasePair *currentPairs = pairsBuff[pairSwap];

	pc.countBegin("setup solver bodies");
	{
		PfxSetupSolverBodiesParam param;
		param.states = states;
		param.bodies = bodies;
		param.solverBodies = solverBodies;
		param.numRigidBodies = numRigidBodies;
		
		int ret = pfxSetupSolverBodies(param);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxSetupSolverBodies failed %d\n",ret);
	}
	pc.countEnd();

	pc.countBegin("setup contact constraints");
	{
		PfxSetupContactConstraintsParam param;
		param.contactPairs = currentPairs;
		param.numContactPairs = numCurrentPairs;
		param.offsetContactManifolds = contacts;
		param.offsetRigidStates = states;
		param.offsetRigidBodies = bodies;
		param.offsetSolverBodies = solverBodies;
		param.numRigidBodies = numRigidBodies;
		param.timeStep = timeStep;
		param.separateBias = separateBias;
		
		int ret = pfxSetupContactConstraints(param);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxSetupJointConstraints failed %d\n",ret);
	}
	pc.countEnd();

	pc.countBegin("setup joint constraints");
	{
		PfxSetupJointConstraintsParam param;
		param.jointPairs = jointPairs;
		param.numJointPairs = numJoints;
		param.offsetJoints = joints;
		param.offsetRigidStates = states;
		param.offsetRigidBodies = bodies;
		param.offsetSolverBodies = solverBodies;
		param.numRigidBodies = numRigidBodies;
		param.timeStep = timeStep;

		for(int i=0;i<numJoints;i++) {
			pfxUpdateJointPairs(jointPairs[i],i,joints[i],states[joints[i].m_rigidBodyIdA],states[joints[i].m_rigidBodyIdB]);
		}

		int ret = pfxSetupJointConstraints(param);
		if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxSetupJointConstraints failed %d\n",ret);
	}
	pc.countEnd();

	pc.countBegin("solve constraints");
	{
		PfxSolveConstraintsParam param;
		param.workBytes = pfxGetWorkBytesOfSolveConstraints(numRigidBodies,numCurrentPairs,numJoints);
		param.workBuff = pool.allocate(param.workBytes);
		param.contactPairs = currentPairs;
		param.numContactPairs = numCurrentPairs;
		param.offsetContactManifolds = contacts;
		param.jointPairs = jointPairs;
		param.numJointPairs = numJoints;
		param.offsetJoints = joints;
		param.offsetRigidStates = states;
		param.offsetSolverBodies = solverBodies;
		param.numRigidBodies = numRigidBodies;
		param.iteration = iteration;

		if (peSolverEnabled)
		{
			int ret = pfxSolveConstraints(param);
			if(ret != SCE_PFX_OK) SCE_PFX_PRINTF("pfxSolveConstraints failed %d\n",ret);
		}
		pool.deallocate(param.workBuff);
	}
	pc.countEnd();

	//pc.printCount();
}

void integrate()
{
	PfxUpdateRigidStatesParam param;
	param.states = states;
	param.bodies = bodies;
	param.numRigidBodies = numRigidBodies;
	param.timeStep = timeStep;
	
	pfxUpdateRigidStates(param);
}

void physics_simulate()
{
	BT_PROFILE("physics_simulate");

	PfxPerfCounter pc;

	for(int i=1;i<numRigidBodies;i++) {
		pfxApplyExternalForce(states[i],bodies[i],bodies[i].getMass()*PfxVector3(0.0f,-9.8f,0.0f),PfxVector3(0.0f),timeStep);
	}
	
//	perf_push_marker("broadphase");
//	pc.countBegin("broadphase");

	{
		BT_PROFILE("broadphase");
		broadphase();
	}
//	
//pc.countEnd();
//	perf_pop_marker();
	
//	perf_push_marker("collision");
//	pc.countBegin("collision");
	{
		BT_PROFILE("collision");
		collision();
	}
//	pc.countEnd();
//	perf_pop_marker();
	
//	perf_push_marker("solver");
//	pc.countBegin("solver");
	{
		BT_PROFILE("constraintSolver");
		constraintSolver();
		if (!peSolverEnabled)
			BulletConstraintSolver();
			

	}
//	pc.countEnd();
//	perf_pop_marker();
	
//	perf_push_marker("integrate");
//	pc.countBegin("integrate");
	{
		BT_PROFILE("integrate");
		integrate();
	}
//	pc.countEnd();
//	perf_pop_marker();
	
	frame++;
	
	if (0)////if(frame%100 == 0)
	{
		float broadphaseTime = pc.getCountTime(0);
		float collisionTime  = pc.getCountTime(2);
		float solverTime     = pc.getCountTime(4);
		float integrateTime  = pc.getCountTime(6);
		SCE_PFX_PRINTF("#pairs = %d, #contacts = %d\n", curNumPairs, curTotalContacts);

		SCE_PFX_PRINTF("frame %3d broadphase %.2f collision %.2f solver %.2f integrate %.2f | total %.2f\n",frame,
			broadphaseTime,collisionTime,solverTime,integrateTime,
			broadphaseTime+collisionTime+solverTime+integrateTime);


	}
}

///////////////////////////////////////////////////////////////////////////////
// Create Scene

void createBrick(int id,const PfxVector3 &pos,const PfxQuat &rot,const PfxVector3 &boxSize,PfxFloat mass)
{
	PfxBox box(boxSize);
	PfxShape shape;
	shape.reset();
	shape.setBox(box);
	collidables[id].reset();
	collidables[id].addShape(shape);
	collidables[id].finish();
	bodies[id].reset();
	bodies[id].setRestitution(0.0f);
	bodies[id].setMass(mass);
	bodies[id].setInertia(pfxCalcInertiaBox(boxSize,mass));
	states[id].reset();
	states[id].setPosition(pos);
	states[id].setOrientation(rot);
	states[id].setMotionType(kPfxMotionTypeActive);
	states[id].setRigidBodyId(id);
}

void createStack(const PfxVector3 &offsetPosition,int stackSize,const PfxVector3 &boxSize)
{
	PfxFloat bodyMass = 0.5f;
    
	PfxFloat diffX = boxSize[0] * 1.0f;
	PfxFloat diffY = boxSize[1] * 1.0f;
	PfxFloat diffZ = boxSize[2] * 1.0f;
    
	PfxVector3 pos(0.0f, diffY, 0.0f);
    
    for(int i=0;i<stackSize;i++) {
        createBrick(numRigidBodies++,offsetPosition+pos,PfxQuat::identity(),boxSize,bodyMass);
    	pos[1] += (diffY * 2.0f);
    }
	
}


void createWall(const PfxVector3 &offsetPosition,int stackSize,const PfxVector3 &boxSize)
{
	PfxFloat bodyMass = 0.5f;

	PfxFloat diffX = boxSize[0] * 1.02f;
	PfxFloat diffY = boxSize[1] * 1.02f;
	PfxFloat diffZ = boxSize[2] * 1.02f;

	PfxFloat offset = -stackSize * (diffZ * 2.0f) * 0.5f;
	PfxVector3 pos(0.0f, diffY, 0.0f);

	while(stackSize) {
		for(int i=0;i<stackSize;i++) {
			pos[2] = offset + (PfxFloat)i * (diffZ * 2.0f);
		
			createBrick(numRigidBodies++,offsetPosition+pos,PfxQuat::identity(),boxSize,bodyMass);
		}
		offset += diffZ;
		pos[1] += (diffY * 2.0f);
		stackSize--;
	}
}

void createPyramid(const PfxVector3 &offsetPosition,int stackSize,const PfxVector3 &boxSize)
{
	PfxFloat space = 0.0001f;
	PfxVector3 pos(0.0f, boxSize[1], 0.0f);

	PfxFloat diffX = boxSize[0] * 1.02f;
	PfxFloat diffY = boxSize[1] * 1.02f;
	PfxFloat diffZ = boxSize[2] * 1.02f;

	PfxFloat offsetX = -stackSize * (diffX * 2.0f + space) * 0.5f;
	PfxFloat offsetZ = -stackSize * (diffZ * 2.0f + space) * 0.5f;
	while(stackSize) {
		for(int j=0;j<stackSize;j++) {
			pos[2] = offsetZ + (PfxFloat)j * (diffZ * 2.0f + space);
			for(int i=0;i<stackSize;i++) {
				pos[0] = offsetX + (PfxFloat)i * (diffX * 2.0f + space);
				createBrick(numRigidBodies++,offsetPosition+pos,PfxQuat::identity(),boxSize,1.0f);
			}
		}
		offsetX += diffX;
		offsetZ += diffZ;
		pos[1] += (diffY * 2.0f + space);
		stackSize--;
	}
}

void createTowerCircle(const PfxVector3 &offsetPosition,int stackSize,int rotSize,const PfxVector3 &boxSize)
{
	PfxFloat radius = 1.3f * rotSize * boxSize[0] / SCE_PFX_PI;

	// create active boxes
	PfxQuat rotY = PfxQuat::identity();
	PfxFloat posY = boxSize[1];

	for(int i=0;i<stackSize;i++) {
		for(int j=0;j<rotSize;j++) {
			createBrick(numRigidBodies++,offsetPosition+rotate(rotY,PfxVector3(0.0f , posY, radius)),rotY,boxSize,0.5f);

			rotY *= PfxQuat::rotationY(SCE_PFX_PI/(rotSize*0.5f));
		}

		posY += boxSize[1] * 2.0f;
		rotY *= PfxQuat::rotationY(SCE_PFX_PI/(PfxFloat)rotSize);
	}
}

void createScenePrimitives()
{
	// sphere
	{
		int id = numRigidBodies++;
		PfxSphere sphere(1.0f);
		PfxShape shape;
		shape.reset();
		shape.setSphere(sphere);
		collidables[id].reset();
		collidables[id].addShape(shape);
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(1.0f);
		bodies[id].setInertia(pfxCalcInertiaSphere(1.0f,1.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(-5.0f,5.0f,0.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}

	// box
	{
		int id = numRigidBodies++;
		PfxBox box(1.0f,1.0f,1.0f);
		PfxShape shape;
		shape.reset();
		shape.setBox(box);
		collidables[id].reset();
		collidables[id].addShape(shape);
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(1.0f);
		bodies[id].setInertia(pfxCalcInertiaBox(PfxVector3(1.0f),1.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(0.0f,5.0f,5.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}

	// capsule
	{
		int id = numRigidBodies++;
		PfxCapsule capsule(1.5f,0.5f);
		PfxShape shape;
		shape.reset();
		shape.setCapsule(capsule);
		collidables[id].reset();
		collidables[id].addShape(shape);
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(2.0f);
		bodies[id].setInertia(pfxCalcInertiaCylinderX(2.0f,0.5f,2.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(5.0f,5.0f,0.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}

	// cylinder
	{
		int id = numRigidBodies++;
		PfxCylinder cylinder(0.5f,1.5f);
		PfxShape shape;
		shape.reset();
		shape.setCylinder(cylinder);
		collidables[id].reset();
		collidables[id].addShape(shape);
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(3.0f);
		bodies[id].setInertia(pfxCalcInertiaCylinderX(0.5f,1.5f,3.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(0.0f,10.0f,0.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}

	// convex mesh
	{
		PfxCreateConvexMeshParam param;

		param.verts = BarrelVtx;
		param.numVerts = BarrelVtxCount;
		param.vertexStrideBytes = sizeof(float)*6;

		param.triangles = BarrelIdx;
		param.numTriangles = BarrelIdxCount/3;
		param.triangleStrideBytes = sizeof(unsigned short)*3;

		PfxInt32 ret = pfxCreateConvexMesh(gConvex,param);
		if(ret != SCE_PFX_OK) {
			SCE_PFX_PRINTF("Can't create gConvex mesh.\n");
		}

		int id = numRigidBodies++;
		PfxShape shape;
		shape.reset();
		shape.setConvexMesh(&gConvex);
		collidables[id].reset();
		collidables[id].addShape(shape);
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(3.0f);
		bodies[id].setInertia(pfxCalcInertiaSphere(1.0f,1.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(0.0f,15.0f,0.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}

	// combined primitives
	{
		int id = numRigidBodies++;

		//E Both shapes and incides buffer have to be kept when creating a combined shape.
		static PfxShape shapes[3];
		PfxUInt16 shapeIds[3]={0,1,2};
		collidables[id].reset(shapes,shapeIds,3);
		{
			PfxBox box(0.5f,0.5f,1.5f);
			PfxShape shape;
			shape.reset();
			shape.setBox(box);
			shape.setOffsetPosition(PfxVector3(-2.0f,0.0f,0.0f));
			collidables[id].addShape(shape);
		}
		{
			PfxBox box(0.5f,1.5f,0.5f);
			PfxShape shape;
			shape.reset();
			shape.setBox(box);
			shape.setOffsetPosition(PfxVector3(2.0f,0.0f,0.0f));
			collidables[id].addShape(shape);
		}
		{
			PfxCapsule cap(1.5f,0.5f);
			PfxShape shape;
			shape.reset();
			shape.setCapsule(cap);
			collidables[id].addShape(shape);
		}
		collidables[id].finish();
		bodies[id].reset();
		bodies[id].setMass(3.0f);
		bodies[id].setInertia(pfxCalcInertiaBox(PfxVector3(2.5f,1.0f,1.0f),3.0f));
		states[id].reset();
		states[id].setPosition(PfxVector3(0.0f,5.0f,0.0f));
		states[id].setMotionType(kPfxMotionTypeActive);
		states[id].setRigidBodyId(id);
	}
}

void createSceneJoints()
{
	const int n = 10;

	int startId = numRigidBodies;

	PfxVector3 boxSize(1.0f);
	PfxFloat boxMass = 1.0f;

	for(int i=0;i<n;i++) {
		createBrick(numRigidBodies++,PfxVector3(0,3.0f+i*2.5f*boxSize[1],0),PfxQuat::identity(),boxSize,boxMass);
	}

	for(int i=startId;i<startId+n;i++) {
		PfxRigidState &stateA = states[i];
		PfxRigidState &stateB = states[(i+1)%numRigidBodies];
		PfxVector3 anchor;
		if(i == numRigidBodies-1) {
			anchor = stateA.getPosition() + PfxVector3(0,boxSize[1],0);
		}
		else {
			anchor = ( stateA.getPosition() + stateB.getPosition() ) * 0.5f;
		}

		PfxSwingTwistJointInitParam jparam;
		jparam.anchorPoint = anchor;
		jparam.twistAxis = PfxVector3(0,1,0);

		pfxInitializeSwingTwistJoint(joints[numJoints],stateA,stateB,jparam);
		joints[numJoints].m_constraints[4].m_damping = 0.1f;
		joints[numJoints].m_constraints[5].m_damping = 0.1f;

		pfxUpdateJointPairs(jointPairs[numJoints],numJoints,joints[numJoints],stateA,stateB);

		SCE_PFX_ASSERT(numJoints<NUM_JOINTS);
		numJoints++;
	}

	states[startId].setLinearVelocity(PfxVector3(0,0,5));
	states[startId].setLinearDamping(0.95f);
	states[startId].setAngularDamping(0.95f);
}

void createSceneStacking()
{
       const float cubeSize = 1.0f;
/*
       createPyramid(PfxVector3(-20.0f,0.0f,0.0f),12,PfxVector3(cubeSize,cubeSize,cubeSize));
       createWall(PfxVector3(-2.0f,0.0f,0.0f),12,PfxVector3(cubeSize,cubeSize,cubeSize));
       createWall(PfxVector3(4.0f,0.0f,0.0f),12,PfxVector3(cubeSize,cubeSize,cubeSize));
       createWall(PfxVector3(10.0f,0.0f,0.0f),12,PfxVector3(cubeSize,cubeSize,cubeSize));
       createTowerCircle(PfxVector3(25.0f,0.0f,0.0f),8,24,PfxVector3(cubeSize,cubeSize,cubeSize));
*/
	//createTowerCircle(PfxVector3(0.0f,0.0f,0.0f),48,24,PfxVector3(1));
    
    createStack(PfxVector3(0.0f,0.0,0.0f),2,PfxVector3(cubeSize,cubeSize,cubeSize));
    
}

void createSceneBoxGround()
{
	int id = numRigidBodies++;
	PfxBox box(150.0f,2.5f,150.0f);
	PfxShape shape;
	shape.reset();
	shape.setBox(box);
	collidables[id].reset();
	collidables[id].addShape(shape);
	collidables[id].finish();
	bodies[id].reset();
	states[id].reset();
	states[id].setPosition(PfxVector3(0.0f,-2.5f,0.0f));
	states[id].setMotionType(kPfxMotionTypeFixed);
	states[id].setRigidBodyId(id);
}

void createSceneLandscape()
{
	PfxCreateLargeTriMeshParam param;

	param.verts = LargeMeshVtx;
	param.numVerts = LargeMeshVtxCount;
	param.vertexStrideBytes = sizeof(float)*6;

	param.triangles = LargeMeshIdx;
	param.numTriangles = LargeMeshIdxCount/3;
	param.triangleStrideBytes = sizeof(unsigned short)*3;

	if(gLargeMesh.m_numIslands > 0) {
		pfxReleaseLargeTriMesh(gLargeMesh);
	}

	PfxInt32 ret = pfxCreateLargeTriMesh(gLargeMesh,param);
	if(ret != SCE_PFX_OK) {
		SCE_PFX_PRINTF("Can't create large mesh.\n");
	}

	int id = numRigidBodies++;
	PfxShape shape;
	shape.reset();
	shape.setLargeTriMesh(&gLargeMesh);
	collidables[id].reset();
	collidables[id].addShape(shape);
	collidables[id].finish();
	bodies[id].reset();
	states[id].reset();
	states[id].setPosition(PfxVector3(0.0f,-5.0f,0.0f));
	states[id].setOrientation(PfxQuat::rotationX(0.5f)*PfxQuat::rotationY(0.7f));
	states[id].setMotionType(kPfxMotionTypeFixed);
	states[id].setRigidBodyId(id);
}

void physics_create_scene(int sceneId)
{
	const int numScenes = 4;
	int sid = sceneId % numScenes;
	
	numRigidBodies= 0;
	pairSwap = 0;
	numPairs[0] = 0;
	numPairs[1] = 0;
	numContacts = 0;
	numContactIdPool = 0;
	numJoints = 0;
	frame = 0;
	
	switch(sid) {
		case 0: // simple primitives
		createSceneBoxGround();
		createScenePrimitives();
		break;
		
		case 1: // joints
		createSceneBoxGround();
		createSceneJoints();
		break;

		case 2: // stacking
		createSceneBoxGround();
		createSceneStacking();
		break;

		case 3: // landscape
		createSceneLandscape();
		createScenePrimitives();
		break;
	}

	SCE_PFX_PRINTF("----- Size of rigid body buffer ------\n");
	SCE_PFX_PRINTF("                    size *   num = total\n");
	SCE_PFX_PRINTF("PfxRigidState      %5d * %5d = %5d bytes\n",sizeof(PfxRigidState),numRigidBodies,sizeof(PfxRigidState)*numRigidBodies);
	SCE_PFX_PRINTF("PfxRigidBody       %5d * %5d = %5d bytes\n",sizeof(PfxRigidBody),numRigidBodies,sizeof(PfxRigidBody)*numRigidBodies);
	SCE_PFX_PRINTF("PfxCollidable      %5d * %5d = %5d bytes\n",sizeof(PfxCollidable),numRigidBodies,sizeof(PfxCollidable)*numRigidBodies);
	SCE_PFX_PRINTF("PfxJoint           %5d * %5d = %5d bytes\n",sizeof(PfxJoint),numJoints,sizeof(PfxJoint)*numJoints);
	SCE_PFX_PRINTF("PfxSolverBody      %5d * %5d = %5d bytes\n",sizeof(PfxSolverBody),numRigidBodies,sizeof(PfxSolverBody)*numRigidBodies);
	SCE_PFX_PRINTF("PfxBroadphaseProxy %5d * %5d = %5d bytes\n",sizeof(PfxBroadphaseProxy),numRigidBodies,sizeof(PfxBroadphaseProxy)*numRigidBodies);
	SCE_PFX_PRINTF("PfxContactManifold %5d * %5d = %5d bytes\n",sizeof(PfxContactManifold),NUM_CONTACTS,sizeof(PfxContactManifold)*NUM_CONTACTS);
	SCE_PFX_PRINTF("PfxBroadphasePair  %5d * %5d = %5d bytes\n",sizeof(PfxBroadphasePair),NUM_CONTACTS,sizeof(PfxBroadphasePair)*NUM_CONTACTS);

	int totalBytes = 
		(sizeof(PfxRigidState) + sizeof(PfxRigidBody) + sizeof(PfxCollidable) + sizeof(PfxSolverBody) + sizeof(PfxBroadphaseProxy)) * numRigidBodies +
		(sizeof(PfxContactManifold) + sizeof(PfxBroadphasePair)) * NUM_CONTACTS;
	SCE_PFX_PRINTF("----------------------------------------------------------\n");
	SCE_PFX_PRINTF("Total %5d bytes\n",totalBytes);
}

///////////////////////////////////////////////////////////////////////////////
// Initialize / Finalize Engine

bool physics_init()
{
	return true;
}

void physics_release()
{
}

///////////////////////////////////////////////////////////////////////////////
// Pick

PfxVector3 physics_pick_start(const PfxVector3 &p1,const PfxVector3 &p2)
{
	return PfxVector3(0.0f);
}

void physics_pick_update(const PfxVector3 &p)
{
}

void physics_pick_end()
{
}

///////////////////////////////////////////////////////////////////////////////
// Get Information

int physics_get_num_rigidbodies()
{
	return numRigidBodies;
}

const PfxRigidState& physics_get_state(int id)
{
	return states[id];
}

const PfxRigidBody& physics_get_body(int id)
{
	return bodies[id];
}

const PfxCollidable& physics_get_collidable(int id)
{
	return collidables[id];
}

int physics_get_num_contacts()
{
	return numPairs[pairSwap];
}

const PfxContactManifold &physics_get_contact(int id)
{
	return contacts[pfxGetConstraintId(pairsBuff[pairSwap][id])];
}

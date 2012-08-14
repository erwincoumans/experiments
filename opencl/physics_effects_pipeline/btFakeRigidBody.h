#ifndef _BT_FAKE_RIGIDBODY_H
#define _BT_FAKE_RIGIDBODY_H

#include "LinearMath/btTransform.h"

class btCollisionObject
{
	public:
	btTransform		m_worldTransform;
	btVector3		m_anisotropicFriction;
	int	m_companionId;

	const btTransform& getWorldTransform() const;
	void setWorldTransform(const btTransform& trans);
	bool hasAnisotropicFriction() const;
	const btVector3& getAnisotropicFriction() const;
	int getCompanionId() const;
	void setCompanionId(int id);
};

class btRigidBody : public btCollisionObject
{
	public:
	btMatrix3x3	m_invInertiaWorld;
	btVector3	m_angularFactor;
	btVector3	m_linearFactor;
	btVector3	m_angularVelocity;
	btVector3	m_linearVelocity;
	btVector3	m_accumulatedForce;
	btVector3	m_accumulatedTorque;
	btScalar	m_invMass;

	
	const btVector3&	getTotalForce() const;
	const btVector3&	getTotalTorque() const;
	const btVector3& getAngularFactor() const;
	const btVector3& getLinearFactor() const;
	const btMatrix3x3& getInvInertiaTensorWorld() const;
	btScalar	getInvMass() const;
	btVector3 getVelocityInLocalPoint(const btVector3& pt) const;
	const btVector3& getAngularVelocity() const;
	const btVector3& getLinearVelocity() const;
	void setLinearVelocity(const btVector3&) ;
	void setAngularVelocity(const btVector3&) ;
	static btRigidBody* upcast(btCollisionObject*);
};

#endif //_BT_FAKE_RIGIDBODY_H

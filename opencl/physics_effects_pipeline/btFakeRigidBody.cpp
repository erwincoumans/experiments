
#include "btFakeRigidBody.h"


const btTransform& btCollisionObject::getWorldTransform() const
{
	return m_worldTransform;
}

void btCollisionObject::setWorldTransform(const btTransform& trans)
{
	m_worldTransform = trans;
}

bool btCollisionObject::hasAnisotropicFriction() const
{
	return true;
}
	
int btCollisionObject::getCompanionId() const
{
	return m_companionId;
}
void btCollisionObject::setCompanionId(int id)
{
	m_companionId = id;
}
const btVector3& btCollisionObject::getAnisotropicFriction() const
{
	return m_anisotropicFriction;
}



const btVector3& btRigidBody::getAngularFactor() const
{
	return m_angularFactor;
}


const btVector3& btRigidBody::getLinearFactor() const
{
	return m_linearFactor;
}

const btMatrix3x3& btRigidBody::getInvInertiaTensorWorld() const
{
	return m_invInertiaWorld;
}
	
btScalar	btRigidBody::getInvMass() const
{
	return m_invMass;
}

btVector3 btRigidBody::getVelocityInLocalPoint(const btVector3& pt) const
{
	btAssert(0);
	return btVector3(0,0,0);
}

const btVector3& btRigidBody::getAngularVelocity() const
{
	return m_angularVelocity;
}

const btVector3& btRigidBody::getLinearVelocity() const
{
	return m_linearVelocity;
}

void btRigidBody::setLinearVelocity(const btVector3& lin) 
{
	m_linearVelocity = lin;
}

void btRigidBody::setAngularVelocity(const btVector3& ang) 
{
	m_angularVelocity = ang;
}

btRigidBody* btRigidBody::upcast(btCollisionObject* obj)
{
	return (btRigidBody*)obj;
}

const btVector3&	btRigidBody::getTotalForce() const
{
	return m_accumulatedForce;
}

const btVector3&	btRigidBody::getTotalTorque() const
{
	return m_accumulatedTorque;
}

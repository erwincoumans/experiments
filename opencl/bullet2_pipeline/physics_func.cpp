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

#include "btBulletDynamicsCommon.h"

btDefaultCollisionConfiguration* g_collisionConfiguration=0;
btCollisionDispatcher* g_dispatcher=0;
btDbvtBroadphase* g_broadphase=0;
btSequentialImpulseConstraintSolver* g_solver=0;
btDiscreteDynamicsWorld* g_dynamicsWorld=0;
btAlignedObjectArray<btCollisionShape*> g_collisionShapes;


//E Simulation
//J シミュレーション
bool physics_init()
{
    return true;
}
void physics_release()
{
}

void physics_create_scene(int sceneId)
{
    
    g_collisionConfiguration = new btDefaultCollisionConfiguration();
    g_dispatcher = new      btCollisionDispatcher(g_collisionConfiguration);
    g_broadphase = new btDbvtBroadphase();
    g_solver = new btSequentialImpulseConstraintSolver;
    g_dynamicsWorld = new btDiscreteDynamicsWorld(g_dispatcher,g_broadphase,g_solver,g_collisionConfiguration);
    g_dynamicsWorld->setGravity(btVector3(0,-10,0));
    
    ///create a few basic rigid bodies
    btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(50.),btScalar(50.),btScalar(50.)));
    g_collisionShapes.push_back(groundShape);
    
    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(0,-50,0));

    {
        btScalar mass(0.);
        
        //rigidbody is dynamic if and only if mass is non zero, otherwise static
        bool isDynamic = (mass != 0.f);
        
        btVector3 localInertia(0,0,0);
        if (isDynamic)
            groundShape->calculateLocalInertia(mass,localInertia);
        
        //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
        btDefaultMotionState* myMotionState = new btDefaultMotionState(groundTransform);
        btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,groundShape,localInertia);
        btRigidBody* body = new btRigidBody(rbInfo);
        
        //add the body to the dynamics world
        g_dynamicsWorld->addRigidBody(body);
    }
    
    {
        btCollisionShape* colShape = new btBoxShape(btVector3(1,1,1));
        g_collisionShapes.push_back(colShape);
        
        /// Create Dynamic Objects
        btTransform startTransform;
        startTransform.setIdentity();
        
        btScalar        mass(1.f);
        
        //rigidbody is dynamic if and only if mass is non zero, otherwise static
        bool isDynamic = (mass != 0.f);
        
        btVector3 localInertia(0,0,0);
        if (isDynamic)
            colShape->calculateLocalInertia(mass,localInertia);
        
        for(int j = 0;j<10;j++)
        {
            startTransform.setOrigin(btVector3(
                                                       btScalar(0),
                                                       btScalar(1+2.0*j),
                                                       btScalar(0.0)));
            
            //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
            btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
            btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,colShape,localInertia);
            btRigidBody* body = new btRigidBody(rbInfo);
            body->setActivationState(DISABLE_DEACTIVATION);
            
            g_dynamicsWorld->addRigidBody(body);
        }
    }
 
}

void physics_simulate()
{
    float deltaTime = 1./60.f;
    g_dynamicsWorld->stepSimulation(deltaTime,0);
}


//E Change parameters
//J パラメータの取得
int physics_get_num_rigidbodies()
{
    return g_dynamicsWorld->getNumCollisionObjects();
}
class btCollisionObject* physics_get_collision_object(int objectIndex)
{
    return g_dynamicsWorld->getCollisionObjectArray()[objectIndex];
}

int physics_get_num_contacts()
{
    return 0;
}
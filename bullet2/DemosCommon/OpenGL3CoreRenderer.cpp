
#include "OpenGL3CoreRenderer.h"
#include "../../rendering/rendertest/GLInstancingRenderer.h"
#include "../../rendering/rendertest/ShapeData.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletCollision/CollisionShapes/btBoxShape.h"


OpenGL3CoreRenderer::OpenGL3CoreRenderer()
{
	int maxNumObjects = 16384;
	m_instanceRenderer = new GLInstancingRenderer(maxNumObjects);
}
OpenGL3CoreRenderer::~OpenGL3CoreRenderer()
{
	delete m_instanceRenderer;
}

void OpenGL3CoreRenderer::init()
{
	m_instanceRenderer->InitShaders();
}



void OpenGL3CoreRenderer::reshape(int w, int h)
{
}
void OpenGL3CoreRenderer::keyboardCallback(unsigned char key)
{
}


//very incomplete conversion from physics to graphics
void graphics_from_physics(GLInstancingRenderer& renderer, bool syncTransformsOnly, const btDiscreteDynamicsWorld* world)
{

    int cubeShapeIndex  = -1;
	int strideInBytes = sizeof(float)*9;
    
	if (!syncTransformsOnly)
	{
		int numVertices = sizeof(cube_vertices)/strideInBytes;
		int numIndices = sizeof(cube_indices)/sizeof(int);
		cubeShapeIndex = renderer.registerShape(&cube_vertices[0],numVertices,cube_indices,numIndices);
	}
    

    
	int numColObj = world->getNumCollisionObjects();
    int curGraphicsIndex = 0;
    
    for (int i=0;i<numColObj;i++)
    {
		btCollisionObject* colObj = world->getCollisionObjectArray()[i];
		
        btVector3 pos = colObj->getWorldTransform().getOrigin();
        btQuaternion orn = colObj->getWorldTransform().getRotation();
        
        float position[4] = {pos.getX(),pos.getY(),pos.getZ(),0.f};
        float orientation[4] = {orn.getX(),orn.getY(),orn.getZ(),orn.getW()};
        float color[4] = {0,0,0,1};
        btVector3 localScaling = colObj->getCollisionShape()->getLocalScaling();
        
       
        if (colObj->isStaticOrKinematicObject())
        {
            color[0]=1.f;
        }else
        {
            color[1]=1.f;
        }
        
        switch (colObj->getCollisionShape()->getShapeType())
        {
            case BOX_SHAPE_PROXYTYPE:
            {
                btBoxShape* box = (btBoxShape*)colObj->getCollisionShape();

                btVector3 halfExtents = box->getHalfExtentsWithMargin();
                
                float cubeScaling[4] = {halfExtents.getX(),halfExtents.getY(), halfExtents.getZ(),1};
                
                if (!syncTransformsOnly)
                {
                    renderer.registerGraphicsInstance(cubeShapeIndex,position,orientation,color,cubeScaling);
                }
                else
                {
                    renderer.writeSingleInstanceTransformToCPU(position,orientation,curGraphicsIndex);
                
                }
                
                curGraphicsIndex++;
            }
            break;
                
            default:
                break;
        }
        //convert it now!
    }
	
}



void OpenGL3CoreRenderer::renderPhysicsWorld(const btDiscreteDynamicsWorld* world)
{
	//sync changes from physics world to render world	
	//for now, we don't deal with adding/removing objects to the world during the simulation, to keep the rendererer simpler


	m_instanceRenderer->writeTransforms();
	static bool syncOnly=false;
	graphics_from_physics(*m_instanceRenderer,syncOnly,world);
	syncOnly= true;

	//render
	 m_instanceRenderer->RenderScene();
}


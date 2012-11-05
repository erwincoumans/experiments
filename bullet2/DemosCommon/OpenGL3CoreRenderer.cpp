
#include "OpenGL3CoreRenderer.h"
#include "../../rendering/rendertest/GLInstancingRenderer.h"
#include "../../rendering/rendertest/ShapeData.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletCollision/CollisionShapes/btBoxShape.h"


OpenGL3CoreRenderer::OpenGL3CoreRenderer()
{
	int maxNumObjects = 128*1024;
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

struct GraphicsVertex
{
	float xyzw[4];
	float normal[3];
	float uv[2];
};
struct GraphicsShape
{
	const float*	m_vertices;
	int				m_numvertices;
	const int*		m_indices;
	int				m_numIndices;
	float			m_scaling[4];
};



GraphicsShape* createGraphicsShapeFromConvexHull(const btConvexPolyhedron* utilPtr)
{
	
	btAlignedObjectArray<GraphicsVertex>* vertices = new btAlignedObjectArray<GraphicsVertex>;
	{
		int numVertices = utilPtr->m_vertices.size();
		int numIndices = 0;
		btAlignedObjectArray<int>* indicesPtr = new btAlignedObjectArray<int>;
		for (int f=0;f<utilPtr->m_faces.size();f++)
		{
			const btFace& face = utilPtr->m_faces[f];
			btVector3 normal(face.m_plane[0],face.m_plane[1],face.m_plane[2]);
			if (face.m_indices.size()>2)
			{
				
				GraphicsVertex vtx;
				const btVector3& orgVertex = utilPtr->m_vertices[face.m_indices[0]];
				vtx.xyzw[0] = orgVertex[0];vtx.xyzw[1] = orgVertex[1];vtx.xyzw[2] = orgVertex[2];vtx.xyzw[3] = 0.f;
				vtx.normal[0] = normal[0];vtx.normal[1] = normal[1];vtx.normal[2] = normal[2];
				vtx.uv[0] = 0.5f;vtx.uv[1] = 0.5f;
				int newvtxindex0 = vertices->size();
				vertices->push_back(vtx);
			
				for (int j=1;j<face.m_indices.size()-1;j++)
				{
					indicesPtr->push_back(newvtxindex0);
					{
						GraphicsVertex vtx;
						const btVector3& orgVertex = utilPtr->m_vertices[face.m_indices[j]];
						vtx.xyzw[0] = orgVertex[0];vtx.xyzw[1] = orgVertex[1];vtx.xyzw[2] = orgVertex[2];vtx.xyzw[3] = 0.f;
						vtx.normal[0] = normal[0];vtx.normal[1] = normal[1];vtx.normal[2] = normal[2];
						vtx.uv[0] = 0.5f;vtx.uv[1] = 0.5f;
						int newvtxindexj = vertices->size();
						vertices->push_back(vtx);
						indicesPtr->push_back(newvtxindexj);
					}

					{
						GraphicsVertex vtx;
						const btVector3& orgVertex = utilPtr->m_vertices[face.m_indices[j+1]];
						vtx.xyzw[0] = orgVertex[0];vtx.xyzw[1] = orgVertex[1];vtx.xyzw[2] = orgVertex[2];vtx.xyzw[3] = 0.f;
						vtx.normal[0] = normal[0];vtx.normal[1] = normal[1];vtx.normal[2] = normal[2];
						vtx.uv[0] = 0.5f;vtx.uv[1] = 0.5f;
						int newvtxindexj1 = vertices->size();
						vertices->push_back(vtx);
						indicesPtr->push_back(newvtxindexj1);
					}
				}
			}
		}
		
		
		GraphicsShape* gfxShape = new GraphicsShape;
		gfxShape->m_vertices = &vertices->at(0).xyzw[0];
		gfxShape->m_numvertices = vertices->size();
		gfxShape->m_indices = &indicesPtr->at(0);
		gfxShape->m_numIndices = indicesPtr->size();
		for (int i=0;i<4;i++)
			gfxShape->m_scaling[i] = 1;//bake the scaling into the vertices 
		return gfxShape;
	}
}


//very incomplete conversion from physics to graphics
void graphics_from_physics(GLInstancingRenderer& renderer, bool syncTransformsOnly, int numObjects, btCollisionObject** colObjArray)
{
	///@todo: we need to sort the objects based on collision shape type, so we can share instances

    
	int strideInBytes = sizeof(float)*9;
    
	int prevGraphicsShapeIndex  = -1;
	btCollisionShape* prevShape = 0;

    
	int numColObj = numObjects;
    int curGraphicsIndex = 0;
    
    for (int i=0;i<numColObj;i++)
    {
		btCollisionObject* colObj = colObjArray[i];
		
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
        
		if (!syncTransformsOnly)
		{
			if (prevShape != colObj->getCollisionShape())
			if (colObj->getCollisionShape()->isPolyhedral())
			{
				btPolyhedralConvexShape* polyShape = (btPolyhedralConvexShape*)colObj->getCollisionShape();
				const btConvexPolyhedron* pol = polyShape->getConvexPolyhedron();
				GraphicsShape* gfxShape = createGraphicsShapeFromConvexHull(pol);

				prevGraphicsShapeIndex = renderer.registerShape(&gfxShape->m_vertices[0],gfxShape->m_numvertices,gfxShape->m_indices,gfxShape->m_numIndices);
				prevShape = colObj->getCollisionShape();
			}
		}
    
		if (colObj->getCollisionShape()->isPolyhedral())
		{
				btPolyhedralConvexShape* polyShape = (btPolyhedralConvexShape*)colObj->getCollisionShape();
				const btVector3& localScaling = polyShape ->getLocalScaling();

				float cubeScaling[4] = {localScaling.getX(),localScaling.getY(), localScaling.getZ(),1};
                
                if (!syncTransformsOnly)
                {
                    renderer.registerGraphicsInstance(prevGraphicsShapeIndex,position,orientation,color,cubeScaling);
                }
                else
                {
                    renderer.writeSingleInstanceTransformToCPU(position,orientation,curGraphicsIndex);
                
                }
                
                curGraphicsIndex++;

		}

       
    }
	
}



void OpenGL3CoreRenderer::renderPhysicsWorld(int numObjects, btCollisionObject** colObjArray, bool syncOnly)
{
	//sync changes from physics world to render world	
	//for now, we don't deal with adding/removing objects to the world during the simulation, to keep the rendererer simpler


	m_instanceRenderer->writeTransforms();
	
	graphics_from_physics(*m_instanceRenderer,syncOnly,numObjects, colObjArray);
	

	//render
	 m_instanceRenderer->RenderScene();
}



#include "OpenGL3CoreRenderer.h"
#include "../../rendering/rendertest/GLInstancingRenderer.h"
#include "../../rendering/rendertest/ShapeData.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"


#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"
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

GraphicsShape* createGraphicsShapeFromConcaveMesh(const btBvhTriangleMeshShape* trimesh)
{
	
	btAlignedObjectArray<GraphicsVertex>* vertices = new btAlignedObjectArray<GraphicsVertex>;
	btAlignedObjectArray<int>* indicesPtr = new btAlignedObjectArray<int>;

	const btStridingMeshInterface* meshInterface = trimesh->getMeshInterface();

	btVector3 trimeshScaling(1,1,1);
	for (int partId=0;partId<meshInterface->getNumSubParts();partId++)
	{
		
		const unsigned char *vertexbase = 0;
		int numverts = 0;
		PHY_ScalarType type = PHY_INTEGER;
		int stride = 0;
		const unsigned char *indexbase = 0;
		int indexstride = 0;
		int numfaces = 0;
		PHY_ScalarType indicestype = PHY_INTEGER;
		//PHY_ScalarType indexType=0;

		btVector3 triangleVerts[3];
		meshInterface->getLockedReadOnlyVertexIndexBase(&vertexbase,numverts,	type,stride,&indexbase,indexstride,numfaces,indicestype,partId);
		btVector3 aabbMin,aabbMax;

		for (int triangleIndex = 0 ; triangleIndex < numfaces;triangleIndex++)
		{
			unsigned int* gfxbase = (unsigned int*)(indexbase+triangleIndex*indexstride);

			for (int j=2;j>=0;j--)
			{

				int graphicsindex = indicestype==PHY_SHORT?((unsigned short*)gfxbase)[j]:gfxbase[j];
				if (type == PHY_FLOAT)
				{
					float* graphicsbase = (float*)(vertexbase+graphicsindex*stride);
					triangleVerts[j] = btVector3(
						graphicsbase[0]*trimeshScaling.getX(),
						graphicsbase[1]*trimeshScaling.getY(),
						graphicsbase[2]*trimeshScaling.getZ());
				}
				else
				{
					double* graphicsbase = (double*)(vertexbase+graphicsindex*stride);
					triangleVerts[j] = btVector3( btScalar(graphicsbase[0]*trimeshScaling.getX()), 
						btScalar(graphicsbase[1]*trimeshScaling.getY()), 
						btScalar(graphicsbase[2]*trimeshScaling.getZ()));
				}
			}
			btVector3 normal = (triangleVerts[2]-triangleVerts[0]).cross(triangleVerts[1]-triangleVerts[0]);
			normal.normalize();

			GraphicsVertex vtx0,vtx1,vtx2;
			vtx0.xyzw[0] = triangleVerts[0].getX();
			vtx0.xyzw[1] = triangleVerts[0].getY();
			vtx0.xyzw[2] = triangleVerts[0].getZ();
			vtx0.xyzw[3] = 0;
			vtx0.uv[0] = 0.5f;
			vtx0.uv[1] = 0.5f;
			vtx0.normal[0] = normal[0];
			vtx0.normal[1] = normal[1];
			vtx0.normal[2] = normal[2];

			vtx1.xyzw[0] = triangleVerts[1].getX();
			vtx1.xyzw[1] = triangleVerts[1].getY();
			vtx1.xyzw[2] = triangleVerts[1].getZ();
			vtx1.xyzw[3] = 0;
			vtx1.uv[0] = 0.5f;
			vtx1.uv[1] = 0.5f;
			vtx1.normal[0] = normal[0];
			vtx1.normal[1] = normal[1];
			vtx1.normal[2] = normal[2];

			vtx2.xyzw[0] = triangleVerts[2].getX();
			vtx2.xyzw[1] = triangleVerts[2].getY();
			vtx2.xyzw[2] = triangleVerts[2].getZ();
			vtx2.xyzw[3] = 0;
			vtx2.uv[0] = 0.5f;
			vtx2.uv[1] = 0.5f;
			vtx2.normal[0] = normal[0];
			vtx2.normal[1] = normal[1];
			vtx2.normal[2] = normal[2];

//			triangleVerts[1]
//			triangleVerts[1]
//			triangleVerts[2]
			vertices->push_back(vtx0);
			vertices->push_back(vtx1);
			vertices->push_back(vtx2);
			indicesPtr->push_back(indicesPtr->size());
			indicesPtr->push_back(indicesPtr->size());
			indicesPtr->push_back(indicesPtr->size());
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
			{
				if (colObj->getCollisionShape()->isPolyhedral())
				{
					btPolyhedralConvexShape* polyShape = (btPolyhedralConvexShape*)colObj->getCollisionShape();
					const btConvexPolyhedron* pol = polyShape->getConvexPolyhedron();
					GraphicsShape* gfxShape = createGraphicsShapeFromConvexHull(pol);

					prevGraphicsShapeIndex = renderer.registerShape(&gfxShape->m_vertices[0],gfxShape->m_numvertices,gfxShape->m_indices,gfxShape->m_numIndices);
					prevShape = colObj->getCollisionShape();
				} else
				{
					if (colObj->getCollisionShape()->getShapeType()==TRIANGLE_MESH_SHAPE_PROXYTYPE)
					{
						btBvhTriangleMeshShape* trimesh = (btBvhTriangleMeshShape*) colObj->getCollisionShape();
						GraphicsShape* gfxShape = createGraphicsShapeFromConcaveMesh(trimesh);
						prevGraphicsShapeIndex = renderer.registerShape(&gfxShape->m_vertices[0],gfxShape->m_numvertices,gfxShape->m_indices,gfxShape->m_numIndices);
						prevShape = colObj->getCollisionShape();
					}

				}
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
		} else
		{
			if (colObj->getCollisionShape()->getShapeType()==TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				float cubeScaling[4]={1,1,1,1};
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


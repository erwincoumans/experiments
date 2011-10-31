#include "CustomConvexShape.h"
#include "ConvexHeightFieldShape.h"
#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"


CustomConvexShape::CustomConvexShape(const btScalar* points,int numPoints, int stride)
:btConvexHullShape(points,numPoints,stride)
{
	
	initializePolyhedralFeatures();
	int numFaces= m_polyhedron->m_faces.size();
	float4* eqn = new float4[numFaces];
	for (int i=0;i<numFaces;i++)
	{
		eqn[i].x = m_polyhedron->m_faces[i].m_plane[0];
		eqn[i].y = m_polyhedron->m_faces[i].m_plane[1];
		eqn[i].z = m_polyhedron->m_faces[i].m_plane[2];
		eqn[i].w = m_polyhedron->m_faces[i].m_plane[3];
	}
	
	m_ConvexHeightField = new ConvexHeightField(eqn,numFaces);

}

CustomConvexShape::~CustomConvexShape()
{
	delete m_ConvexHeightField;
}
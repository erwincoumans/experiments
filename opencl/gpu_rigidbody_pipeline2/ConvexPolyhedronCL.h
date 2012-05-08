#ifndef CONVEX_POLYHEDRON_CL
#define CONVEX_POLYHEDRON_CL

#include "LinearMath/btTransform.h"

struct btGpuFace
{
	int m_indexOffset;
	int m_numIndices;
//	btAlignedObjectArray<int>	m_indices;
//	btAlignedObjectArray<int>	m_connectedFaces;
	btScalar	m_plane[4];
};

struct ConvexPolyhedronCL
{

	int	m_faceOffset;
	int m_numFaces;

	//btAlignedObjectArray<btFace>	m_faces;

	int	m_numVertices;
	int m_vertexOffset;

	int	m_uniqueEdgesOffset;
	int	m_numUniqueEdges;

	btVector3		m_localCenter;
	btVector3		m_extents;
	btScalar		m_radius;
	btVector3		mC;
	btVector3		mE;

	inline void project(const btTransform& trans, const btVector3& dir, const btAlignedObjectArray<btVector3>& vertices, btScalar& min, btScalar& max) const
	{
		min = FLT_MAX;
		max = -FLT_MAX;
		int numVerts = m_numVertices;
		for(int i=0;i<numVerts;i++)
		{
			btVector3 pt = trans * vertices[m_vertexOffset+i];
			btScalar dp = pt.dot(dir);
			if(dp < min)	min = dp;
			if(dp > max)	max = dp;
		}
		if(min>max)
		{
			btScalar tmp = min;
			min = max;
			max = tmp;
		}
	}

};

#endif //CONVEX_POLYHEDRON_CL
#ifndef CONVEX_POLYHEDRON_CL
#define CONVEX_POLYHEDRON_CL

#include "LinearMath/btTransform.h"

struct btFace
{
	btAlignedObjectArray<int>	m_indices;
//	btAlignedObjectArray<int>	m_connectedFaces;
	btScalar	m_plane[4];
};

struct ConvexPolyhedronCL
{
	
	btAlignedObjectArray<btVector3>	m_vertices;
	btAlignedObjectArray<btFace>	m_faces;
	btAlignedObjectArray<btVector3> m_uniqueEdges;

	btVector3		m_localCenter;
	btVector3		m_extents;
	btScalar		m_radius;
	btVector3		mC;
	btVector3		mE;

	inline void project(const btTransform& trans, const btVector3& dir, btScalar& min, btScalar& max) const
	{
		min = FLT_MAX;
		max = -FLT_MAX;
		int numVerts = m_vertices.size();
		for(int i=0;i<numVerts;i++)
		{
			btVector3 pt = trans * m_vertices[i];
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
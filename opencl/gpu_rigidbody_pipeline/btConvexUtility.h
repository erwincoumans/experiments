
#ifndef _BT_CONVEX_UTILITY_H
#define _BT_CONVEX_UTILITY_H

#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btVector3.h"

struct btFace
{
	btAlignedObjectArray<int>	m_indices;
//	btAlignedObjectArray<int>	m_connectedFaces;
	btScalar	m_plane[4];
};

class btConvexUtility
{
	public:
		
	btAlignedObjectArray<btVector3>	m_vertices;
	btAlignedObjectArray<btFace>	m_faces;
	
	bool	initializePolyhedralFeatures(const btAlignedObjectArray<btVector3>& orgVertices, bool mergeCoplanarTriangles);
		
};
#endif
	
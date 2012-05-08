typedef unsigned int u32;

typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;

	u32 m_shapeIdx;
	u32 m_shapeType;
	
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} BodyData;


typedef struct  
{
	float4		m_localCenter;
	float4		m_extents;
	float4		mC;
	float4		mE;
	float			m_radius;
	
	int	m_faceOffset;
	int m_numFaces;

	int	m_numVertices;
	int m_vertexOffset;

	int	m_uniqueEdgesOffset;
	int	m_numUniqueEdges;

} ConvexPolyhedronCL;

/*	inline void project(const btTransform& trans, const btVector3& dir, const btAlignedObjectArray<btVector3>& vertices, btScalar& min, btScalar& max) const
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
*/

/*

static bool TestSepAxis(const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, const btTransform& transA,const btTransform& transB, const btVector3& sep_axis, const btAlignedObjectArray<btVector3> vertices,btScalar& depth)
{
	btScalar Min0,Max0;
	btScalar Min1,Max1;
	hullA.project(transA,sep_axis,vertices, Min0, Max0);
	hullB.project(transB, sep_axis,vertices, Min1, Max1);

	if(Max0<Min1 || Max1<Min0)
		return false;

	btScalar d0 = Max0 - Min1;
	assert(d0>=0.0f);
	btScalar d1 = Max1 - Min0;
	assert(d1>=0.0f);
	depth = d0<d1 ? d0:d1;
	return true;
}
*/





// work-in-progress
__kernel void   findSeparatingAxisKernel( __global const int2* pairs, 
																					__global const BodyData* rigidBodies, 
																					__global const ConvexPolyhedronCL* convexShapes, 
																					__global volatile float4* separatingNormals, 
																					int numPairs)
{
		
	int i = get_global_id(0);
	if (i<numPairs)
	{
		int shapeIndexA = rigidBodies[pairs[i].x].m_shapeIdx;
		int numFacesA = convexShapes[shapeIndexA].m_numFaces;
	 
		separatingNormals[i] = (float4)((float)pairs[i].x,(float)pairs[i].y,(float)numFacesA,(float)i);
	}
}


/*

static bool findSeparatingAxis(	const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, 
	const btTransform& transA,const btTransform& transB, 
	const btAlignedObjectArray<btVector3>& vertices, 
	const btAlignedObjectArray<btVector3>& uniqueEdges, 
	const btAlignedObjectArray<btGpuFace>& faces,
	const btAlignedObjectArray<int>& indices,
	btVector3& sep)
{


//@todo: we could still enable this, even without internal object
#ifdef TEST_INTERNAL_OBJECTS
	const btVector3 c0 = transA * hullA.m_localCenter;
	const btVector3 c1 = transB * hullB.m_localCenter;
	const btVector3 DeltaC2 = c0 - c1;
#endif

	btScalar dmin = FLT_MAX;
	int curPlaneTests=0;

	int numFacesA = hullA.m_numFaces;
	// Test normals from hullA
	for(int i=0;i<numFacesA;i++)
	{
		const btVector3 Normal(faces[hullA.m_faceOffset+i].m_plane[0], faces[hullA.m_faceOffset+i].m_plane[1], faces[hullA.m_faceOffset+i].m_plane[2]);
		const btVector3 faceANormalWS = transA.getBasis() * Normal;
#ifdef TEST_INTERNAL_OBJECTS
		if (DeltaC2.dot(faceANormalWS)<0)
			continue;
#endif //TEST_INTERNAL_OBJECTS

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB, DeltaC2, faceANormalWS, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		if(!TestSepAxis( hullA, hullB, transA,transB, faceANormalWS, vertices,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = faceANormalWS;
		}
	}

	int numFacesB = hullB.m_numFaces;
	// Test normals from hullB
	for(int i=0;i<numFacesB;i++)
	{
		const btVector3 Normal(faces[hullB.m_faceOffset+i].m_plane[0], faces[hullB.m_faceOffset+i].m_plane[1], faces[hullB.m_faceOffset+i].m_plane[2]);
		const btVector3 WorldNormal = transB.getBasis() * Normal;
#ifdef TEST_INTERNAL_OBJECTS
		if (DeltaC2.dot(WorldNormal)<0)
			continue;
#endif

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, WorldNormal, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		if(!TestSepAxis(hullA, hullB,transA,transB, WorldNormal,vertices,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = WorldNormal;
		}
	}

	btVector3 edgeAstart,edgeAend,edgeBstart,edgeBend;

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA.m_numUniqueEdges;e0++)
	{
		const btVector3 edge0 = uniqueEdges[hullA.m_uniqueEdgesOffset+e0];
		const btVector3 WorldEdge0 = transA.getBasis() * edge0;
		for(int e1=0;e1<hullB.m_numUniqueEdges;e1++)
		{
			const btVector3 edge1 = uniqueEdges[hullB.m_uniqueEdgesOffset+e1];
			const btVector3 WorldEdge1 = transB.getBasis() * edge1;

			btVector3 Cross = WorldEdge0.cross(WorldEdge1);
			curEdgeEdge++;
			if(!IsAlmostZero(Cross))
			{
				Cross = Cross.normalize();
				if (DeltaC2.dot(Cross)<0)
					continue;


#ifdef TEST_INTERNAL_OBJECTS
				gExpectedNbTests++;
				if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, Cross, hullA, hullB, dmin))
					continue;
				gActualNbTests++;
#endif

				btScalar dist;
				if(!TestSepAxis( hullA, hullB, transA,transB, Cross, vertices,dist))
					return false;

				if(dist<dmin)
				{
					dmin = dist;
					sep = Cross;
				}
			}
		}

	}

	const btVector3 deltaC = transB.getOrigin() - transA.getOrigin();
	if((deltaC.dot(sep))>0.0f)
		sep = -sep;

	return true;
}
*/

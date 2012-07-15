
//keep this enum in sync with the CPU version (in AdlCollisionShape.h)
#define SHAPE_CONVEX_HULL 3
#define SHAPE_CONCAVE_TRIMESH 5

typedef unsigned int u32;

///keep this in sync with btCollidable.h
typedef struct
{
	int m_shapeType;
	int m_shapeIndex;
	
} btCollidableGpu;

typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;

	u32 m_collidableIdx;
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

typedef struct 
{
	union
	{
		float4	m_min;
		float   m_minElems[4];
		int			m_minIndices[4];
	};
	union
	{
		float4	m_max;
		float   m_maxElems[4];
		int			m_maxIndices[4];
	};
} btAabbCL;

typedef struct
{
	float4 m_plane;
	int m_indexOffset;
	int m_numIndices;
} btGpuFace;

#define make_float4 (float4)

__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.xyz,0.f);
	float4 b1 = make_float4(b.xyz,0.f);
	return dot(a1, b1);
}

__inline
float4 fastNormalize4(float4 v)
{
	return fast_normalize(v);
}


///////////////////////////////////////
//	Quaternion
///////////////////////////////////////

typedef float4 Quaternion;

__inline
Quaternion qtMul(Quaternion a, Quaternion b);

__inline
Quaternion qtNormalize(Quaternion in);

__inline
float4 qtRotate(Quaternion q, float4 vec);

__inline
Quaternion qtInvert(Quaternion q);




__inline
Quaternion qtMul(Quaternion a, Quaternion b)
{
	Quaternion ans;
	ans = cross3( a, b );
	ans += a.w*b+b.w*a;
//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w*b.w - dot3F4(a, b);
	return ans;
}

__inline
Quaternion qtNormalize(Quaternion in)
{
	return fastNormalize4(in);
//	in /= length( in );
//	return in;
}
__inline
float4 qtRotate(Quaternion q, float4 vec)
{
	Quaternion qInv = qtInvert( q );
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = qtMul(qtMul(q,vcpy),qInv);
	return out;
}

__inline
Quaternion qtInvert(Quaternion q)
{
	return (Quaternion)(-q.xyz, q.w);
}

__inline
float4 qtInvRotate(const Quaternion q, float4 vec)
{
	return qtRotate( qtInvert( q ), vec );
}

__inline
float4 transform(const float4* p, const float4* translation, const Quaternion* orientation)
{
	return qtRotate( *orientation, *p ) + (*translation);
}



__inline
float4 normalize3(const float4 a)
{
	float4 n = make_float4(a.x, a.y, a.z, 0.f);
	return fastNormalize4( n );
}

inline void project(__global const ConvexPolyhedronCL* hull,  const float4 pos, const float4 orn, 
const float4* dir, __global const float4* vertices, float* min, float* max)
{
	min[0] = FLT_MAX;
	max[0] = -FLT_MAX;
	int numVerts = hull->m_numVertices;

	const float4 localDir = qtInvRotate(orn,*dir);
	float offset = dot(pos,*dir);
	for(int i=0;i<numVerts;i++)
	{
		float dp = dot(vertices[hull->m_vertexOffset+i],localDir);
		if(dp < min[0])	
			min[0] = dp;
		if(dp > max[0])	
			max[0] = dp;
	}
	if(min[0]>max[0])
	{
		float tmp = min[0];
		min[0] = max[0];
		max[0] = tmp;
	}
	min[0] += offset;
	max[0] += offset;
}


inline bool TestSepAxis(__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA,const float4 ornA,
	const float4 posB,const float4 ornB,
	float4* sep_axis, __global const float4* vertices,float* depth)
{
	float Min0,Max0;
	float Min1,Max1;
	project(hullA,posA,ornA,sep_axis,vertices, &Min0, &Max0);
	project(hullB,posB,ornB, sep_axis,vertices, &Min1, &Max1);

	if(Max0<Min1 || Max1<Min0)
		return false;

	float d0 = Max0 - Min1;
	float d1 = Max1 - Min0;
	*depth = d0<d1 ? d0:d1;
	return true;
}



inline bool IsAlmostZero(const float4 v)
{
	if(fabs(v.x)>1e-6f || fabs(v.y)>1e-6f || fabs(v.z)>1e-6f)	
		return false;
	return true;
}




bool findSeparatingAxis(	__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	__global const float4* vertices, 
	__global const float4* uniqueEdges, 
	__global const btGpuFace* faces,
	__global const int*  indices,
	__global volatile float4* sep,
	float* dmin)
{
	int i = get_global_id(0);

	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;
	
	int curPlaneTests=0;

	{
		int numFacesA = hullA->m_numFaces;
		// Test normals from hullA
		for(int i=0;i<numFacesA;i++)
		{
			const float4 normal = faces[hullA->m_faceOffset+i].m_plane;
			float4 faceANormalWS = qtRotate(ornA,normal);
	
			if (dot3F4(DeltaC2,faceANormalWS)<0)
				continue;
	
			curPlaneTests++;
	
			float d;
			if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,&faceANormalWS, vertices,&d))
				return false;
	
			if(d<*dmin)
			{
				*dmin = d;
				*sep = faceANormalWS;
			}
		}
	}

	
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}




bool findSeparatingAxisEdgeEdge(	__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	__global const float4* vertices, 
	__global const float4* uniqueEdges, 
	__global const btGpuFace* faces,
	__global const int*  indices,
	__global volatile float4* sep,
	float* dmin)
{
	int i = get_global_id(0);

	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;

	int curPlaneTests=0;

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA->m_numUniqueEdges;e0++)
	{
		const float4 edge0 = uniqueEdges[hullA->m_uniqueEdgesOffset+e0];
		float4 edge0World = qtRotate(ornA,edge0);

		for(int e1=0;e1<hullB->m_numUniqueEdges;e1++)
		{
			const float4 edge1 = uniqueEdges[hullB->m_uniqueEdgesOffset+e1];
			float4 edge1World = qtRotate(ornB,edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero(crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(DeltaC2,crossje)<0)
					continue;

				float dist;
				bool result = true;
				{
					float Min0,Max0;
					float Min1,Max1;
					project(hullA,posA,ornA,&crossje,vertices, &Min0, &Max0);
					project(hullB,posB,ornB,&crossje,vertices, &Min1, &Max1);
				
					if(Max0<Min1 || Max1<Min0)
						result = false;
				
					float d0 = Max0 - Min1;
					float d1 = Max1 - Min0;
					dist = d0<d1 ? d0:d1;
					result = true;

				}
				

				if(dist<*dmin)
				{
					*dmin = dist;
					*sep = crossje;
				}
			}
		}

	}

	
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}




// work-in-progress
__kernel void   findSeparatingAxisKernel( __global const int2* pairs, 
																					__global const BodyData* rigidBodies, 
																					__global const btCollidableGpu* collidables,
																					__global const ConvexPolyhedronCL* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* uniqueEdges,
																					__global const btGpuFace* faces,
																					__global const int* indices,
																					__global const btAabbCL* aabbs,
																					__global volatile float4* separatingNormals,
																					__global volatile int* hasSeparatingAxis,
																					__global int4* concavePairsOut,
																					__global volatile int* numConcavePairsOut,
																					int numPairs,
																					int maxnumConcavePairsCapacity
																					)
{

	int i = get_global_id(0);
	if (i<numPairs)
	{
		int bodyIndexA = pairs[i].x;
		int bodyIndexB = pairs[i].y;

		if ((rigidBodies[bodyIndexA].m_shapeType==SHAPE_CONCAVE_TRIMESH) ||(rigidBodies[bodyIndexB].m_shapeType==SHAPE_CONCAVE_TRIMESH))
		{
			int pairIdx = atomic_inc(numConcavePairsOut);
			if (pairIdx<maxnumConcavePairsCapacity)
			{
				concavePairsOut[pairIdx].x = bodyIndexA;
				concavePairsOut[pairIdx].y = bodyIndexB;
				concavePairsOut[pairIdx].z = 1;
				concavePairsOut[pairIdx].w = 2;
			}
			//todo
			hasSeparatingAxis[i] = 0;
			return;
		}		

		if ((rigidBodies[bodyIndexA].m_shapeType!=SHAPE_CONVEX_HULL) ||(rigidBodies[bodyIndexB].m_shapeType!=SHAPE_CONVEX_HULL))
		{
			hasSeparatingAxis[i] = 0;
			return;
		}
			

//once the broadphase avoids static-static pairs, we can remove this test
		if ((rigidBodies[bodyIndexA].m_invMass==0) &&(rigidBodies[bodyIndexB].m_invMass==0))
		{
			hasSeparatingAxis[i] = 0;
			return;
		}

		int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
		int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;

		int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
		int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
		
		int numFacesA = convexShapes[shapeIndexA].m_numFaces;

		float dmin = FLT_MAX;

		float4 posA = rigidBodies[bodyIndexA].m_pos;
		posA.w = 0.f;
		float4 posB = rigidBodies[bodyIndexB].m_pos;
		posB.w = 0.f;
		float4 c0local = convexShapes[shapeIndexA].m_localCenter;
		float4 ornA = rigidBodies[bodyIndexA].m_quat;
		float4 c0 = transform(&c0local, &posA, &ornA);
		float4 c1local = convexShapes[shapeIndexB].m_localCenter;
		float4 ornB =rigidBodies[bodyIndexB].m_quat;
		float4 c1 = transform(&c1local,&posB,&ornB);
		const float4 DeltaC2 = c0 - c1;
		
		bool sepA = findSeparatingAxis(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,
																								rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,
																								DeltaC2,
																								vertices,uniqueEdges,faces,
																								indices,&separatingNormals[i],&dmin);
		hasSeparatingAxis[i] = 4;
		if (!sepA)
		{
			hasSeparatingAxis[i] = 0;
		} else
		{
			bool sepB = findSeparatingAxis(	&convexShapes[shapeIndexB],&convexShapes[shapeIndexA],rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,
																									rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,
																									DeltaC2,
																									vertices,uniqueEdges,faces,
																									indices,&separatingNormals[i],&dmin);

			if (!sepB)
			{
				hasSeparatingAxis[i] = 0;
			} else
			{
				bool sepEE = findSeparatingAxisEdgeEdge(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,
																									rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,
																									DeltaC2,
																									vertices,uniqueEdges,faces,
																									indices,&separatingNormals[i],&dmin);
				if (!sepEE)
				{
					hasSeparatingAxis[i] = 0;
				} else
				{
					hasSeparatingAxis[i] = 1;
				}
			}
		}
		
	}
}
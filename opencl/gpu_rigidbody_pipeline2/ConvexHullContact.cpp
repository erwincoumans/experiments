/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


///This file was written by Erwin Coumans
///Separating axis rest based on work from Pierre Terdiman, see
///And contact clipping based on work from Simon Hobbs

#include "ConvexHullContact.h"

#include "ConvexPolyhedronCL.h"

typedef btAlignedObjectArray<btVector3> btVertexArray;
#include "LinearMath/btQuickprof.h"

#include <float.h> //for FLT_MAX
#include "../basic_initialize/btOpenCLUtils.h"
#include "../broadphase_benchmark/btLauncherCL.h"
//#include "AdlQuaternion.h"

GpuSatCollision::GpuSatCollision(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_findSeparatingAxisKernel(0)
{
	char* src = 0;
	cl_int errNum=0;

	if (1)
	{
		cl_program satProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,src,&errNum,"","../../opencl/gpu_rigidbody_pipeline2/sat.cl");
		btAssert(errNum==CL_SUCCESS);

		m_findSeparatingAxisKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,src, "findSeparatingAxisKernel",&errNum,satProg );
		btAssert(errNum==CL_SUCCESS);
	}

}

GpuSatCollision::~GpuSatCollision()
{
	if (m_findSeparatingAxisKernel)
		clReleaseKernel(m_findSeparatingAxisKernel);

	
}

int gExpectedNbTests=0;
int gActualNbTests = 0;
bool gUseInternalObject = true;

// Clips a face to the back of a plane
static void clipFace(const btVertexArray& pVtxIn, btVertexArray& ppVtxOut, const btVector3& planeNormalWS,btScalar planeEqWS)
{
	
	int ve;
	btScalar ds, de;
	int numVerts = pVtxIn.size();
	if (numVerts < 2)
		return;

	btVector3 firstVertex=pVtxIn[pVtxIn.size()-1];
	btVector3 endVertex = pVtxIn[0];
	
	ds = planeNormalWS.dot(firstVertex)+planeEqWS;

	for (ve = 0; ve < numVerts; ve++)
	{
		endVertex=pVtxIn[ve];

		de = planeNormalWS.dot(endVertex)+planeEqWS;

		if (ds<0)
		{
			if (de<0)
			{
				// Start < 0, end < 0, so output endVertex
				ppVtxOut.push_back(endVertex);
			}
			else
			{
				// Start < 0, end >= 0, so output intersection
				ppVtxOut.push_back( 	firstVertex.lerp(endVertex,btScalar(ds * 1.f/(ds - de))));
			}
		}
		else
		{
			if (de<0)
			{
				// Start >= 0, end < 0 so output intersection and end
				ppVtxOut.push_back(firstVertex.lerp(endVertex,btScalar(ds * 1.f/(ds - de))));
				ppVtxOut.push_back(endVertex);
			}
		}
		firstVertex = endVertex;
		ds = de;
	}
}

inline void project(const ConvexPolyhedronCL& hull,  const float4& pos, const float4& orn, const float4& dir, const btAlignedObjectArray<btVector3>& vertices, btScalar& min, btScalar& max)
{
	min = FLT_MAX;
	max = -FLT_MAX;
	int numVerts = hull.m_numVertices;

	const float4 localDir = qtInvRotate(orn,(float4&)dir);

	btScalar offset = dot3F4(pos,dir);

	for(int i=0;i<numVerts;i++)
	{
		//btVector3 pt = trans * vertices[m_vertexOffset+i];
		//btScalar dp = pt.dot(dir);
		btScalar dp = dot3F4((float4&)vertices[hull.m_vertexOffset+i],localDir);
		//btAssert(dp==dpL);
		if(dp < min)	min = dp;
		if(dp > max)	max = dp;
	}
	if(min>max)
	{
		btScalar tmp = min;
		min = max;
		max = tmp;
	}
	min += offset;
	max += offset;
}


static bool TestSepAxis(const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, 
	const float4& posA,const float4& ornA,
	const float4& posB,const float4& ornB,
	const float4& sep_axis, const btAlignedObjectArray<btVector3> vertices,btScalar& depth)
{
	btScalar Min0,Max0;
	btScalar Min1,Max1;
	project(hullA,posA,ornA,sep_axis,vertices, Min0, Max0);
	project(hullB,posB,ornB, sep_axis,vertices, Min1, Max1);

	if(Max0<Min1 || Max1<Min0)
		return false;

	btScalar d0 = Max0 - Min1;
	assert(d0>=0.0f);
	btScalar d1 = Max1 - Min0;
	assert(d1>=0.0f);
	depth = d0<d1 ? d0:d1;
	return true;
}



static int gActualSATPairTests=0;

inline bool IsAlmostZero(const btVector3& v)
{
	if(fabsf(v.x())>1e-6 || fabsf(v.y())>1e-6 || fabsf(v.z())>1e-6)	return false;
	return true;
}

#ifdef TEST_INTERNAL_OBJECTS

inline void BoxSupport(const btScalar extents[3], const btScalar sv[3], btScalar p[3])
{
	// This version is ~11.000 cycles (4%) faster overall in one of the tests.
//	IR(p[0]) = IR(extents[0])|(IR(sv[0])&SIGN_BITMASK);
//	IR(p[1]) = IR(extents[1])|(IR(sv[1])&SIGN_BITMASK);
//	IR(p[2]) = IR(extents[2])|(IR(sv[2])&SIGN_BITMASK);
	p[0] = sv[0] < 0.0f ? -extents[0] : extents[0];
	p[1] = sv[1] < 0.0f ? -extents[1] : extents[1];
	p[2] = sv[2] < 0.0f ? -extents[2] : extents[2];
}

static void InverseTransformPoint3x3(btVector3& out, const btVector3& in, const btTransform& tr)
{
	const btMatrix3x3& rot = tr.getBasis();
	const btVector3& r0 = rot[0];
	const btVector3& r1 = rot[1];
	const btVector3& r2 = rot[2];

	const btScalar x = r0.x()*in.x() + r1.x()*in.y() + r2.x()*in.z();
	const btScalar y = r0.y()*in.x() + r1.y()*in.y() + r2.y()*in.z();
	const btScalar z = r0.z()*in.x() + r1.z()*in.y() + r2.z()*in.z();

	out.setValue(x, y, z);
}

static bool TestInternalObjects( const btTransform& trans0, const btTransform& trans1, const btVector3& delta_c, const btVector3& axis, const ConvexPolyhedronCL& convex0, const ConvexPolyhedronCL& convex1, btScalar dmin)
{
	const btScalar dp = delta_c.dot(axis);

	btVector3 localAxis0;
	InverseTransformPoint3x3(localAxis0, axis,trans0);
	btVector3 localAxis1;
	InverseTransformPoint3x3(localAxis1, axis,trans1);

	btScalar p0[3];
	BoxSupport(convex0.m_extents, localAxis0, p0);
	btScalar p1[3];
	BoxSupport(convex1.m_extents, localAxis1, p1);

	const btScalar Radius0 = p0[0]*localAxis0.x() + p0[1]*localAxis0.y() + p0[2]*localAxis0.z();
	const btScalar Radius1 = p1[0]*localAxis1.x() + p1[1]*localAxis1.y() + p1[2]*localAxis1.z();

	const btScalar MinRadius = Radius0>convex0.m_radius ? Radius0 : convex0.m_radius;
	const btScalar MaxRadius = Radius1>convex1.m_radius ? Radius1 : convex1.m_radius;

	const btScalar MinMaxRadius = MaxRadius + MinRadius;
	const btScalar d0 = MinMaxRadius + dp;
	const btScalar d1 = MinMaxRadius - dp;

	const btScalar depth = d0<d1 ? d0:d1;
	if(depth>dmin)
		return false;
	return true;
}
#endif //TEST_INTERNAL_OBJECTS


static bool findSeparatingAxis(	const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, 
	const float4& posA1,
	const float4& ornA,
	const float4& posB1,
	const float4& ornB,
	const btAlignedObjectArray<btVector3>& vertices, 
	const btAlignedObjectArray<btVector3>& uniqueEdges, 
	const btAlignedObjectArray<btGpuFace>& faces,
	const btAlignedObjectArray<int>& indices,
	btVector3& sep)
{
	BT_PROFILE("findSeparatingAxis");
	gActualSATPairTests++;
	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;
//#ifdef TEST_INTERNAL_OBJECTS
	float4 c0local = (float4&)hullA.m_localCenter;
	float4 c0 = transform(c0local, posA, ornA);
	float4 c1local = (float4&)hullB.m_localCenter;
	float4 c1 = transform(c1local,posB,ornB);
	const float4 DeltaC2 = c0 - c1;
//#endif

	btScalar dmin = FLT_MAX;
	int curPlaneTests=0;

	int numFacesA = hullA.m_numFaces;
	// Test normals from hullA
	for(int i=0;i<numFacesA;i++)
	{
		const float4& normal = (float4&)faces[hullA.m_faceOffset+i].m_plane;
		float4 faceANormalWS = qtRotate(ornA,normal);

		if (dot3F4(DeltaC2,faceANormalWS)<0)
			continue;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB, DeltaC2, faceANormalWS, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,faceANormalWS, vertices,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = (btVector3&)faceANormalWS;
		}
	}

	int numFacesB = hullB.m_numFaces;
	// Test normals from hullB
	for(int i=0;i<numFacesB;i++)
	{
		float4 normal = (float4&)faces[hullB.m_faceOffset+i].m_plane;
		const float4 WorldNormal = qtRotate(ornB, normal);

		if (dot3F4(DeltaC2,WorldNormal)<0)
			continue;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, WorldNormal, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		if(!TestSepAxis(hullA, hullB,posA,ornA,posB,ornB,WorldNormal,vertices,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = (btVector3&)WorldNormal;
		}
	}

	btVector3 edgeAstart,edgeAend,edgeBstart,edgeBend;

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA.m_numUniqueEdges;e0++)
	{
		const float4& edge0 = (float4&) uniqueEdges[hullA.m_uniqueEdgesOffset+e0];
		float4 edge0World = qtRotate(ornA,(float4&)edge0);

		for(int e1=0;e1<hullB.m_numUniqueEdges;e1++)
		{
			const btVector3 edge1 = uniqueEdges[hullB.m_uniqueEdgesOffset+e1];
			float4 edge1World = qtRotate(ornB,(float4&)edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero((btVector3&)crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(DeltaC2,crossje)<0)
					continue;


#ifdef TEST_INTERNAL_OBJECTS
				gExpectedNbTests++;
				if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, Cross, hullA, hullB, dmin))
					continue;
				gActualNbTests++;
#endif

				btScalar dist;
				if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,crossje, vertices,dist))
					return false;

				if(dist<dmin)
				{
					dmin = dist;
					sep = (btVector3&)crossje;
				}
			}
		}

	}

	const float4 deltaC = posB - posA;
	if((dot3F4(deltaC,(float4&)sep))>0.0f)
		sep = -sep;

	return true;
}

static void	clipFaceAgainstHull(const btVector3& separatingNormal, const ConvexPolyhedronCL& hullA,  
	const btTransform& transA, btVertexArray& worldVertsB1, 
	const btScalar minDist, btScalar maxDist,
	const btAlignedObjectArray<btVector3>& vertices,
	const btAlignedObjectArray<btGpuFace>& faces,
	const btAlignedObjectArray<int>& indices,
	ContactResult& resultOut)
{

	btVertexArray worldVertsB2;
	btVertexArray* pVtxIn = &worldVertsB1;
	btVertexArray* pVtxOut = &worldVertsB2;
	pVtxOut->reserve(pVtxIn->size());

	int closestFaceA=-1;
	{
		btScalar dmin = FLT_MAX;
		for(int face=0;face<hullA.m_numFaces;face++)
		{
			const btVector3 Normal(
				faces[hullA.m_faceOffset+face].m_plane.x, 
				faces[hullA.m_faceOffset+face].m_plane.y, 
				faces[hullA.m_faceOffset+face].m_plane.z);
			const btVector3 faceANormalWS = transA.getBasis() * Normal;
		
			btScalar d = faceANormalWS.dot(separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
			}
		}
	}
	if (closestFaceA<0)
		return;

	const btGpuFace& polyA = faces[hullA.m_faceOffset+closestFaceA];

		// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numContacts = pVtxIn->size();
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const btVector3& a = vertices[hullA.m_vertexOffset+indices[polyA.m_indexOffset+e0]];
		const btVector3& b = vertices[hullA.m_vertexOffset+indices[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
		const btVector3 edge0 = a - b;
		const btVector3 WorldEdge0 = transA.getBasis() * edge0;
		btVector3 worldPlaneAnormal1 = transA.getBasis()* btVector3(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z);

		btVector3 planeNormalWS1 = -WorldEdge0.cross(worldPlaneAnormal1);//.cross(WorldEdge0);
		btVector3 worldA1 = transA*a;
		btScalar planeEqWS1 = -worldA1.dot(planeNormalWS1);
		
//int otherFace=0;
#ifdef BLA1
		int otherFace = polyA.m_connectedFaces[e0];
		btVector3 localPlaneNormal (hullA.m_faces[otherFace].m_plane[0],hullA.m_faces[otherFace].m_plane[1],hullA.m_faces[otherFace].m_plane[2]);
		btScalar localPlaneEq = hullA.m_faces[otherFace].m_plane[3];

		btVector3 planeNormalWS = transA.getBasis()*localPlaneNormal;
		btScalar planeEqWS=localPlaneEq-planeNormalWS.dot(transA.getOrigin());
#else 
		btVector3 planeNormalWS = planeNormalWS1;
		btScalar planeEqWS=planeEqWS1;
		
#endif
		//clip face

		clipFace(*pVtxIn, *pVtxOut,planeNormalWS,planeEqWS);
		btSwap(pVtxIn,pVtxOut);
		pVtxOut->resize(0);
	}



//#define ONLY_REPORT_DEEPEST_POINT

	btVector3 point;
	

	// only keep points that are behind the witness face
	{
		btVector3 localPlaneNormal (polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z);
		btScalar localPlaneEq = polyA.m_plane.w;
		btVector3 planeNormalWS = transA.getBasis()*localPlaneNormal;
		btScalar planeEqWS=localPlaneEq-planeNormalWS.dot(transA.getOrigin());
		for (int i=0;i<pVtxIn->size();i++)
		{
			
			btScalar depth = planeNormalWS.dot(pVtxIn->at(i))+planeEqWS;
			if (depth <=minDist)
			{
//				printf("clamped: depth=%f to minDist=%f\n",depth,minDist);
				depth = minDist;
			}

			if (depth <=maxDist)
			{
				btVector3 point = pVtxIn->at(i);
#ifdef ONLY_REPORT_DEEPEST_POINT
				curMaxDist = depth;
#else
#if 0
				if (depth<-3)
				{
					printf("error in btPolyhedralContactClipping depth = %f\n", depth);
					printf("likely wrong separatingNormal passed in\n");
				} 
#endif				
				resultOut.addContactPoint(separatingNormal,point,depth);
#endif
			}
		}
	}
#ifdef ONLY_REPORT_DEEPEST_POINT
	if (curMaxDist<maxDist)
	{
		resultOut.addContactPoint(separatingNormal,point,curMaxDist);
	}
#endif //ONLY_REPORT_DEEPEST_POINT

}


static void	clipHullAgainstHull(const btVector3& separatingNormal1, 
	const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, 
	const btTransform& transA,const btTransform& transB, const btScalar minDist, btScalar maxDist,
	const btAlignedObjectArray<btVector3>& vertices,
	const btAlignedObjectArray<btGpuFace>& faces,
	const btAlignedObjectArray<int>& indices,


	ContactResult& resultOut)
{
	BT_PROFILE("clipHullAgainstHull");
	btVector3 separatingNormal = separatingNormal1.normalized();
	const btVector3 c0 = transA * hullA.m_localCenter;
	const btVector3 c1 = transB * hullB.m_localCenter;
	const btVector3 DeltaC2 = c0 - c1;


	btScalar curMaxDist=maxDist;
	int closestFaceB=-1;
	btScalar dmax = -FLT_MAX;
	{
		for(int face=0;face<hullB.m_numFaces;face++)
		{
			const btVector3 Normal(faces[hullB.m_faceOffset+face].m_plane.x, 
				faces[hullB.m_faceOffset+face].m_plane.y, faces[hullB.m_faceOffset+face].m_plane.z);
			const btVector3 WorldNormal = transB.getBasis() * Normal;
			btScalar d = WorldNormal.dot(separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}

	btVertexArray worldVertsB1;
	btAssert(closestFaceB>=0);
	{
		const btGpuFace& polyB = faces[hullB.m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		worldVertsB1.reserve(numVertices);
		for(int e0=0;e0<numVertices;e0++)
		{
			const btVector3& b = vertices[hullB.m_vertexOffset+indices[polyB.m_indexOffset+e0]];
			worldVertsB1.push_back(transB*b);
		}
	}

	
	if (closestFaceB>=0)
		clipFaceAgainstHull(separatingNormal, hullA, transA,worldVertsB1, minDist, maxDist,vertices,faces,indices,resultOut);

}






#define PARALLEL_SUM(v, n) for(int j=1; j<n; j++) v[0] += v[j];
#define PARALLEL_DO(execution, n) for(int ie=0; ie<n; ie++){execution;}
#define REDUCE_MAX(v, n) {int i=0;\
	for(int offset=0; offset<n; offset++) v[i] = (v[i].y > v[i+offset].y)? v[i]: v[i+offset]; }
#define REDUCE_MIN(v, n) {int i=0;\
	for(int offset=0; offset<n; offset++) v[i] = (v[i].y < v[i+offset].y)? v[i]: v[i+offset]; }

int extractManifold(const float4* p, int nPoints, float4& nearNormal, float4& centerOut, 
					 int contactIdx[4])
{
	if( nPoints == 0 ) return 0;

	nPoints = min2( nPoints, 64 );

	float4 center = make_float4(0.f);
	{
		float4 v[64];
		memcpy( v, p, nPoints*sizeof(float4) );
		PARALLEL_SUM( v, nPoints );
		center = v[0]/(float)nPoints;
	}

	centerOut = center;

	{	//	sample 4 directions
		if( nPoints < 4 )
		{
			for(int i=0; i<nPoints; i++) contactIdx[i] = i;
			return nPoints;
		}

		float4 aVector = p[0] - center;
		float4 u = cross3( nearNormal, aVector );
		float4 v = cross3( nearNormal, u );
		u = normalize3( u );
		v = normalize3( v );

		int idx[4];

		float2 max00 = make_float2(0,FLT_MAX);
		{
			float4 dir0 = u;
			float4 dir1 = -u;
			float4 dir2 = v;
			float4 dir3 = -v;

			//	idx, distance
			{
				{
					int4 a[64];
					for(int ie = 0; ie<nPoints; ie++ )
					{
						float4 f;
						float4 r = p[ie]-center;
						f.x = dot3F4( dir0, r );
						f.y = dot3F4( dir1, r );
						f.z = dot3F4( dir2, r );
						f.w = dot3F4( dir3, r );

						a[ie].x = ((*(u32*)&f.x) & 0xffffff00);
						a[ie].x |= (0xff & ie);

						a[ie].y = ((*(u32*)&f.y) & 0xffffff00);
						a[ie].y |= (0xff & ie);

						a[ie].z = ((*(u32*)&f.z) & 0xffffff00);
						a[ie].z |= (0xff & ie);

						a[ie].w = ((*(u32*)&f.w) & 0xffffff00);
						a[ie].w |= (0xff & ie);
					}

					for(int ie=0; ie<nPoints; ie++)
					{
						a[0].x = (a[0].x > a[ie].x )? a[0].x: a[ie].x;
						a[0].y = (a[0].y > a[ie].y )? a[0].y: a[ie].y;
						a[0].z = (a[0].z > a[ie].z )? a[0].z: a[ie].z;
						a[0].w = (a[0].w > a[ie].w )? a[0].w: a[ie].w;
					}

					idx[0] = (int)a[0].x & 0xff;
					idx[1] = (int)a[0].y & 0xff;
					idx[2] = (int)a[0].z & 0xff;
					idx[3] = (int)a[0].w & 0xff;
				}
			}

			{
				float2 h[64];
				PARALLEL_DO( h[ie] = make_float2((float)ie, p[ie].w), nPoints );
				REDUCE_MIN( h, nPoints );
				max00 = h[0];
			}
		}

		contactIdx[0] = idx[0];
		contactIdx[1] = idx[1];
		contactIdx[2] = idx[2];
		contactIdx[3] = idx[3];

//		if( max00.y < 0.0f )
//			contactIdx[0] = (int)max00.x;

		std::sort( contactIdx, contactIdx+4 );

		return 4;
	}
}

struct ContactAccumulator: public ContactResult
{
		float4							m_normalOnSurfaceB;
		btAlignedObjectArray<float4>	m_closestPointInBs;
		btAlignedObjectArray<btScalar>	m_distances; //negative means penetration !

		ContactAccumulator()
		{

		}
		virtual ~ContactAccumulator() {};

		virtual void setShapeIdentifiersA(int partId0,int index0)
		{
		}
		virtual void setShapeIdentifiersB(int partId1,int index1)
		{
		}

		virtual void addContactPoint(const btVector3& normalOnBInWorld1,const btVector3& pointInWorld,btScalar depth)
		{
			btVector3 normalOnBInWorld = -normalOnBInWorld1;
			
			float4 normalWorld = make_float4(normalOnBInWorld.getX(),normalOnBInWorld.getY(),normalOnBInWorld.getZ(),0);
			float4 pos = make_float4(pointInWorld.getX(),pointInWorld.getY(),pointInWorld.getZ(),depth);
			m_normalOnSurfaceB = normalWorld;
			m_closestPointInBs.push_back(pos);
			m_closestPointInBs[m_closestPointInBs.size()-1].w = depth;

		}
};



void GpuSatCollision::computeConvexConvexContactsHost( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const ChNarrowphase::Config& cfg , 
			const btAlignedObjectArray<ConvexPolyhedronCL>* hostConvexData,
			const btAlignedObjectArray<btVector3>& vertices,
			const btAlignedObjectArray<btVector3>& uniqueEdges,
			const btAlignedObjectArray<btGpuFace>& faces,
			const btAlignedObjectArray<int>& indices)
{
	
	BT_PROFILE("computeConvexConvexContactsHost");

	m_hostPairs.resize(pairs->size());
	if (pairs->size())
	{
		BT_PROFILE("copyToHost(m_hostPairs)");
		pairs->copyToHost(m_hostPairs);
	}
	
	if (contactOut->size())
	{
		BT_PROFILE("copyToHost(m_hostContactOut");
		contactOut->copyToHost(m_hostContactOut);
	}
	btAlignedObjectArray<RigidBodyBase::Body> hostBodyBuf;
	{
		BT_PROFILE("copyToHost(hostBodyBuf");
		bodyBuf->copyToHost(hostBodyBuf);
	}
	btAlignedObjectArray<ChNarrowphase::ShapeData> hostShapeBuf;
	{
		BT_PROFILE("copyToHost(hostShapeBuf");
		shapeBuf->copyToHost(hostShapeBuf);
	}



	btAssert(m_hostPairs.size() == nPairs);
	m_hostContactOut.reserve(nPairs);
	ContactAccumulator resultOut;
	
	btAlignedObjectArray<float4> hostNormals;
	btAlignedObjectArray<int> hostHasSep;


	{
		btOpenCLArray<int> contactCount(m_context, m_queue);
		btOpenCLArray<float4> sepNormals(m_context,m_queue);
		sepNormals.resize(nPairs);
		btOpenCLArray<int> hasSeparatingNormals(m_context,m_queue);
		hasSeparatingNormals.resize(nPairs);

		btOpenCLArray<btVector3> gpuVertices(m_context,m_queue);
		gpuVertices.copyFromHost(vertices);
		btOpenCLArray<btVector3> gpuUniqueEdges(m_context,m_queue);
		gpuUniqueEdges.copyFromHost(uniqueEdges);
		btOpenCLArray<btGpuFace> gpuFaces(m_context,m_queue);	
		gpuFaces.copyFromHost(faces);
		btOpenCLArray<int> gpuIndices(m_context,m_queue);
		gpuIndices.copyFromHost(indices);

		contactCount.push_back(0);
		btOpenCLArray<ConvexPolyhedronCL> convexData(m_context,m_queue);
		convexData.copyFromHost(*hostConvexData);
		//work-in-progress

		clFinish(m_queue);
		if (1)
		{
			BT_PROFILE("findSeparatingAxisKernel");
			btBufferInfoCL bInfo[] = { 
				btBufferInfoCL( pairs->getBufferCL(), true ), 
				btBufferInfoCL( bodyBuf->getBufferCL(),true), 
				btBufferInfoCL( convexData.getBufferCL(),true),
				btBufferInfoCL( gpuVertices.getBufferCL(),true),
				btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				btBufferInfoCL( gpuFaces.getBufferCL(),true),
				btBufferInfoCL( gpuIndices.getBufferCL(),true),


				btBufferInfoCL( sepNormals.getBufferCL()),
				btBufferInfoCL( hasSeparatingNormals.getBufferCL())};
			btLauncherCL launcher(m_queue, m_findSeparatingAxisKernel);
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( nPairs  );
			int num = nPairs;
			launcher.launch1D( num);
			clFinish(m_queue);
		}

		if (sepNormals.size())
		{
			sepNormals.copyToHost(hostNormals);
			hasSeparatingNormals.copyToHost(hostHasSep);
		}
//		printf("hostNormals.size()=%d\n",hostNormals.size());
		//int numPairs = pairCount.at(0);
		
	}
	
	
	//m_hostContactOut.reserve(nPairs);
	m_hostContactOut.resize(nPairs+nContacts);//m_hostContactOut.size()+1);
	int actualContacts = 0;
	for (int i=0;i<nPairs;i++)
	{
		int indexA = m_hostPairs[i].x;
		int indexB = m_hostPairs[i].y;
		int shapeA = hostBodyBuf[indexA].m_shapeIdx;
		int shapeB = hostBodyBuf[indexB].m_shapeIdx;

	
		

		bool validateFindSeparatingAxis = false;
		if (validateFindSeparatingAxis)
		{
		
			btVector3 sepNormalWorldSpace;
			bool foundSepAxis =false;

			BT_PROFILE("findSeparatingAxis");
			foundSepAxis = findSeparatingAxis(
						hostConvexData->at(shapeA), 
						hostConvexData->at(shapeB),
						hostBodyBuf[indexA].m_pos,
						hostBodyBuf[indexA].m_quat,
						hostBodyBuf[indexB].m_pos,
						hostBodyBuf[indexB].m_quat,

						vertices,uniqueEdges,faces,indices,sepNormalWorldSpace);
			if (foundSepAxis != hostHasSep[i])
			{
				printf("not matching boolean with gpu at %d\n",i);
			} 
			if (foundSepAxis && (btFabs(sepNormalWorldSpace[0]-hostNormals[i].x)>1e06f))
			{
				printf("not matching normal %f != %f with gpu at %d\n",sepNormalWorldSpace[0],hostNormals[i].x,i);
			}
			

		}
	

		if (hostHasSep[i])
		{
			BT_PROFILE("clipHullAgainstHull");
	
			btScalar minDist = -1;
			btScalar maxDist = 0.1;

			resultOut.m_closestPointInBs.resize(0);
			resultOut.m_distances.resize(0);
			

			btTransform trA,trB;
		trA.setIdentity();
		trA.setOrigin(btVector3(hostBodyBuf[indexA].m_pos.x,hostBodyBuf[indexA].m_pos.y,hostBodyBuf[indexA].m_pos.z));
		trA.setRotation(btQuaternion(hostBodyBuf[indexA].m_quat.x,hostBodyBuf[indexA].m_quat.y,hostBodyBuf[indexA].m_quat.z,hostBodyBuf[indexA].m_quat.w));
		
		trB.setIdentity();
		trB.setOrigin(btVector3(hostBodyBuf[indexB].m_pos.x,hostBodyBuf[indexB].m_pos.y,hostBodyBuf[indexB].m_pos.z));
		trB.setRotation(btQuaternion(hostBodyBuf[indexB].m_quat.x,hostBodyBuf[indexB].m_quat.y,hostBodyBuf[indexB].m_quat.z,hostBodyBuf[indexB].m_quat.w));


			clipHullAgainstHull((btVector3&)hostNormals[i], 
				hostConvexData->at(shapeA), 
				hostConvexData->at(shapeA),
							trA,
							trB,minDist, maxDist,vertices,faces,indices,resultOut);
			
		}
		bool overlap = resultOut.m_closestPointInBs.size()>0;
		if (overlap)
		{

			BT_PROFILE("overlap");
			float4 centerOut;
			int contactIdx[4]={-1,-1,-1,-1};

			int numPoints = 0;
			
			{
				BT_PROFILE("extractManifold");
				numPoints = extractManifold(&resultOut.m_closestPointInBs[0], resultOut.m_closestPointInBs.size(), resultOut.m_normalOnSurfaceB, centerOut,  contactIdx);
			}
			
			{
				//BT_PROFILE("m_hostContactOut.resize");
				
			}
			Contact4& contact = m_hostContactOut[nContacts];
			contact.m_batchIdx = i;
			contact.m_bodyAPtr = m_hostPairs[i].x;
			contact.m_bodyBPtr = m_hostPairs[i].y;
			contact.m_frictionCoeffCmp = 45874;
			contact.m_restituitionCoeffCmp = 0;
			
			float distance = 0.f;
			for (int p=0;p<numPoints;p++)
			{
				contact.m_worldPos[p] = resultOut.m_closestPointInBs[contactIdx[p]];
				contact.m_worldNormal = resultOut.m_normalOnSurfaceB; 
			}
			contact.m_worldNormal.w = numPoints;
			nContacts++;
		}
	}

	m_hostContactOut.resize(nContacts);

	//nContacts = m_hostContactOut.size();
	{
		BT_PROFILE("copyFromHost(m_hostContactOut");
		contactOut->copyFromHost(m_hostContactOut);
	}
}

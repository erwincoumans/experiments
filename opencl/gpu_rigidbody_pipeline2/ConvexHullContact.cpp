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

#include "satKernels.h"
#include "satClipKernels.h"



GpuSatCollision::GpuSatCollision(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_findSeparatingAxisKernel(0)
{
	
	cl_int errNum=0;

	if (1)
	{
		const char* src = satKernelsCL;
		cl_program satProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,src,&errNum,"","../../opencl/gpu_rigidbody_pipeline2/sat.cl");
		btAssert(errNum==CL_SUCCESS);

		m_findSeparatingAxisKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,src, "findSeparatingAxisKernel",&errNum,satProg );
		btAssert(errNum==CL_SUCCESS);
	}

	if (1)
	{
		const char* srcClip = satClipKernelsCL;
		cl_program satClipContactsProg = btOpenCLUtils::compileCLProgramFromString(m_context,m_device,srcClip,&errNum,"","../../opencl/gpu_rigidbody_pipeline2/satClipHullContacts.cl");
		btAssert(errNum==CL_SUCCESS);

		m_clipHullHullKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipHullHullKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);

		m_extractManifoldAndAddContactKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "extractManifoldAndAddContactKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);

	} else
	{
		m_clipHullHullKernel=0;
		m_extractManifoldAndAddContactKernel = 0;
	}
	

}

GpuSatCollision::~GpuSatCollision()
{
	if (m_findSeparatingAxisKernel)
		clReleaseKernel(m_findSeparatingAxisKernel);

	if (m_clipHullHullKernel)
		clReleaseKernel(m_clipHullHullKernel);
	if (m_extractManifoldAndAddContactKernel)
		clReleaseKernel(m_extractManifoldAndAddContactKernel);
}

int gExpectedNbTests=0;
int gActualNbTests = 0;
bool gUseInternalObject = true;

__inline float4 lerp3(const float4& a,const float4& b, float  t)
{
	return make_float4(	a.x + (b.x - a.x) * t,
						a.y + (b.y - a.y) * t,
						a.z + (b.z - a.z) * t,
						0.f);
}


// Clips a face to the back of a plane, return the number of vertices out, stored in ppVtxOut
int clipFace(const float4* pVtxIn, int numVertsIn, float4& planeNormalWS,float planeEqWS, float4* ppVtxOut)
{
	
	int ve;
	float ds, de;
	int numVertsOut = 0;
	if (numVertsIn < 2)
		return 0;

	float4 firstVertex=pVtxIn[numVertsIn-1];
	float4 endVertex = pVtxIn[0];
	
	ds = dot3F4(planeNormalWS,firstVertex)+planeEqWS;

	for (ve = 0; ve < numVertsIn; ve++)
	{
		endVertex=pVtxIn[ve];

		de = dot3F4(planeNormalWS,endVertex)+planeEqWS;

		if (ds<0)
		{
			if (de<0)
			{
				// Start < 0, end < 0, so output endVertex
				ppVtxOut[numVertsOut++] = endVertex;
			}
			else
			{
				// Start < 0, end >= 0, so output intersection
				ppVtxOut[numVertsOut++] = lerp3(firstVertex, endVertex,(ds * 1.f/(ds - de)) );
			}
		}
		else
		{
			if (de<0)
			{
				// Start >= 0, end < 0 so output intersection and end
				ppVtxOut[numVertsOut++] = lerp3(firstVertex, endVertex,(ds * 1.f/(ds - de)) );
				ppVtxOut[numVertsOut++] = endVertex;
			}
		}
		firstVertex = endVertex;
		ds = de;
	}
	return numVertsOut;
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



int clipFaceAgainstHull(const float4& separatingNormal, const ConvexPolyhedronCL* hullA,  
	const float4& posA, const Quaternion& ornA, float4* worldVertsB1, int numWorldVertsB1,
	float4* worldVertsB2, int capacityWorldVertsB2,
	const float minDist, float maxDist,
	const float4* vertices,
	const btGpuFace* faces,
	const int* indices,
	float4* contactsOut,
	int contactCapacity)
{
	int numContactsOut = 0;

	float4* pVtxIn = worldVertsB1;
	float4* pVtxOut = worldVertsB2;
	
	int numVertsIn = numWorldVertsB1;
	int numVertsOut = 0;

	int closestFaceA=-1;
	{
		float dmin = FLT_MAX;
		for(int face=0;face<hullA->m_numFaces;face++)
		{
			const float4 Normal = make_float4(
				faces[hullA->m_faceOffset+face].m_plane.x, 
				faces[hullA->m_faceOffset+face].m_plane.y, 
				faces[hullA->m_faceOffset+face].m_plane.z,0.f);
			const float4 faceANormalWS = qtRotate(ornA,Normal);
		
			float d = dot3F4(faceANormalWS,separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
			}
		}
	}
	if (closestFaceA<0)
		return numContactsOut;

	btGpuFace polyA = faces[hullA->m_faceOffset+closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numContacts = numWorldVertsB1;
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const float4 a = vertices[hullA->m_vertexOffset+indices[polyA.m_indexOffset+e0]];
		const float4 b = vertices[hullA->m_vertexOffset+indices[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
		const float4 edge0 = a - b;
		const float4 WorldEdge0 = qtRotate(ornA,edge0);
		float4 planeNormalA = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float4 worldPlaneAnormal1 = qtRotate(ornA,planeNormalA);

		float4 planeNormalWS1 = -cross3(WorldEdge0,worldPlaneAnormal1);
		float4 worldA1 = transform(a,posA,ornA);
		float planeEqWS1 = -dot3F4(worldA1,planeNormalWS1);
		
		float4 planeNormalWS = planeNormalWS1;
		float planeEqWS=planeEqWS1;
		
		//clip face
		//clipFace(*pVtxIn, *pVtxOut,planeNormalWS,planeEqWS);
		numVertsOut = clipFace(pVtxIn, numVertsIn, planeNormalWS,planeEqWS, pVtxOut);

		//btSwap(pVtxIn,pVtxOut);
		float4* tmp = pVtxOut;
		pVtxOut = pVtxIn;
		pVtxIn = tmp;
		numVertsIn = numVertsOut;
		numVertsOut = 0;
	}

	
	// only keep points that are behind the witness face
	{
		float4 localPlaneNormal  = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float localPlaneEq = polyA.m_plane.w;
		float4 planeNormalWS = qtRotate(ornA,localPlaneNormal);
		float planeEqWS=localPlaneEq-dot3F4(planeNormalWS,posA);
		for (int i=0;i<numVertsIn;i++)
		{
			float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
			if (depth <=minDist)
			{
				depth = minDist;
			}

			if (depth <=maxDist)
			{
				float4 pointInWorld = pVtxIn[i];
				//resultOut.addContactPoint(separatingNormal,point,depth);
				contactsOut[numContactsOut++] = make_float4(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
			}
		}
	}

	return numContactsOut;
}



static int	clipHullAgainstHull(const float4& separatingNormal, 
	const ConvexPolyhedronCL& hullA, const ConvexPolyhedronCL& hullB, 
	const float4& posA, const Quaternion& ornA,const float4& posB, const Quaternion& ornB, 
	float4* worldVertsB1, float4* worldVertsB2, int capacityWorldVerts,
	const float minDist, float maxDist,
	const float4* vertices,
	const btGpuFace* faces,
	const int* indices,
	float4*	contactsOut,
	int contactCapacity)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;

	BT_PROFILE("clipHullAgainstHull");

	float curMaxDist=maxDist;
	int closestFaceB=-1;
	float dmax = -FLT_MAX;

	{
		//BT_PROFILE("closestFaceB");
		for(int face=0;face<hullB.m_numFaces;face++)
		{
			const float4 Normal = make_float4(faces[hullB.m_faceOffset+face].m_plane.x, 
				faces[hullB.m_faceOffset+face].m_plane.y, faces[hullB.m_faceOffset+face].m_plane.z,0.f);
			const float4 WorldNormal = qtRotate(ornB, Normal);
			float d = dot3F4(WorldNormal,separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}

	
	btAssert(closestFaceB>=0);
	{
		//BT_PROFILE("worldVertsB1");
		const btGpuFace& polyB = faces[hullB.m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4& b = vertices[hullB.m_vertexOffset+indices[polyB.m_indexOffset+e0]];
			worldVertsB1[numWorldVertsB1++] = transform(b,posB,ornB);
		}
	}

	if (closestFaceB>=0)
	{
		//BT_PROFILE("clipFaceAgainstHull");
		numContactsOut = clipFaceAgainstHull((float4&)separatingNormal, &hullA, 
				posA,ornA,
				worldVertsB1,numWorldVertsB1,worldVertsB2,capacityWorldVerts, minDist, maxDist,vertices,
				faces,
				indices,contactsOut,contactCapacity);
	}

	return numContactsOut;
}






#define PARALLEL_SUM(v, n) for(int j=1; j<n; j++) v[0] += v[j];
#define PARALLEL_DO(execution, n) for(int ie=0; ie<n; ie++){execution;}
#define REDUCE_MAX(v, n) {int i=0;\
	for(int offset=0; offset<n; offset++) v[i] = (v[i].y > v[i+offset].y)? v[i]: v[i+offset]; }
#define REDUCE_MIN(v, n) {int i=0;\
	for(int offset=0; offset<n; offset++) v[i] = (v[i].y < v[i+offset].y)? v[i]: v[i+offset]; }

int extractManifold(const float4* p, int nPoints, float4& nearNormal, float4& centerOut,  int contactIdx[4])
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
			for(int i=0; i<nPoints; i++) 
				contactIdx[i] = i;
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

		//does this sort happen on GPU too?
		//std::sort( contactIdx, contactIdx+4 );

		return 4;
	}
}


#define MAX_VERTS 1024


void GpuSatCollision::computeConvexConvexContactsGPUSAT( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const ChNarrowphase::Config& cfg , 
			const btOpenCLArray<ConvexPolyhedronCL>& convexData,
			const btOpenCLArray<btVector3>& gpuVertices,
			const btOpenCLArray<btVector3>& gpuUniqueEdges,
			const btOpenCLArray<btGpuFace>& gpuFaces,
			const btOpenCLArray<int>& gpuIndices)
{
	if (!nPairs)
		return;

	BT_PROFILE("computeConvexConvexContactsGPUSAT");

	btOpenCLArray<float4> sepNormals(m_context,m_queue);
	sepNormals.resize(nPairs);
	btOpenCLArray<int> hasSeparatingNormals(m_context,m_queue);
	hasSeparatingNormals.resize(nPairs);


				btAlignedObjectArray<ConvexPolyhedronCL> hostConvexData;
			btAlignedObjectArray<RigidBodyBase::Body> hostBodyBuf;

	btAlignedObjectArray<float4> hostNormals;
	btAlignedObjectArray<int> hostHasSep;

	bool findSeparatingAxisOnGpu = true;


	{

		

		clFinish(m_queue);
		if (findSeparatingAxisOnGpu)
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
		} else
		{

			{

				BT_PROFILE("copyToHost(convexData)");
				convexData.copyToHost(hostConvexData);
			}

			{
				BT_PROFILE("copyToHost(hostBodyBuf");
				bodyBuf->copyToHost(hostBodyBuf);
			}

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


			btAlignedObjectArray<btVector3> vertices;
		{
			BT_PROFILE("copyToHost(gpuVertices)");
			gpuVertices.copyToHost(vertices);
		}
		btAlignedObjectArray<btGpuFace> faces;
		{
			BT_PROFILE("copyToHost(gpuFaces)");
			gpuFaces.copyToHost(faces);
		}

		btAlignedObjectArray<btVector3> uniqueEdges;
		{
			BT_PROFILE("copyToHost(gpuUniqueEdges)");
			gpuUniqueEdges.copyToHost(uniqueEdges);
		}

		btAlignedObjectArray<int> indices;
		{
			BT_PROFILE("copyToHost(gpuIndices)");
			gpuIndices.copyToHost(indices);
		}


			hostHasSep.resize(nPairs);
			hostNormals.resize(nPairs);

			for (int i=0;i<nPairs;i++)
			{
				int indexA = m_hostPairs[i].x;
				int indexB = m_hostPairs[i].y;
				int shapeA = hostBodyBuf[indexA].m_shapeIdx;
				int shapeB = hostBodyBuf[indexB].m_shapeIdx;


				btVector3 sepNormalWorldSpace;
				bool foundSepAxis =false;

				BT_PROFILE("findSeparatingAxis");
				foundSepAxis = findSeparatingAxis(
							hostConvexData.at(shapeA), 
							hostConvexData.at(shapeB),
							hostBodyBuf[indexA].m_pos,
							hostBodyBuf[indexA].m_quat,
							hostBodyBuf[indexB].m_pos,
							hostBodyBuf[indexB].m_quat,

							vertices,uniqueEdges,faces,indices,sepNormalWorldSpace);

				hostHasSep[i] = foundSepAxis;
				if (foundSepAxis)
					hostNormals[i] = make_float4(sepNormalWorldSpace.getX(),sepNormalWorldSpace.getY(),sepNormalWorldSpace.getZ(),0.f);

			}
		}

//		printf("hostNormals.size()=%d\n",hostNormals.size());
		//int numPairs = pairCount.at(0);
		
	}
	
	bool contactClippingOnGpu = true;
	if (contactClippingOnGpu)
	{
		BT_PROFILE("clipHullHullKernel");

		btOpenCLArray<int> totalContactsOut(m_context, m_queue);
		totalContactsOut.push_back(nContacts);

		btBufferInfoCL bInfo[] = { 
			btBufferInfoCL( pairs->getBufferCL(), true ), 
			btBufferInfoCL( bodyBuf->getBufferCL(),true), 
			btBufferInfoCL( convexData.getBufferCL(),true),
			btBufferInfoCL( gpuVertices.getBufferCL(),true),
			btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
			btBufferInfoCL( gpuFaces.getBufferCL(),true),
			btBufferInfoCL( gpuIndices.getBufferCL(),true),
			btBufferInfoCL( sepNormals.getBufferCL()),
			btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
			btBufferInfoCL( contactOut->getBufferCL()),
			btBufferInfoCL( totalContactsOut.getBufferCL())	
		};
		btLauncherCL launcher(m_queue, m_clipHullHullKernel);
		launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
		launcher.setConst( nPairs  );
		int num = nPairs;
		launcher.launch1D( num);
		clFinish(m_queue);
		
		nContacts = totalContactsOut.at(0);

	} else
	{	
		bool reductionOnGpu = true;

	
		btAlignedObjectArray<btVector3> vertices;
		{
			BT_PROFILE("copyToHost(gpuVertices)");
			gpuVertices.copyToHost(vertices);
		}
		btAlignedObjectArray<btGpuFace> faces;
		{
			BT_PROFILE("copyToHost(gpuFaces)");
			gpuFaces.copyToHost(faces);
		}
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
		
		{
			BT_PROFILE("copyToHost(hostBodyBuf");
			bodyBuf->copyToHost(hostBodyBuf);
		}
		btAlignedObjectArray<ChNarrowphase::ShapeData> hostShapeBuf;
		{
			BT_PROFILE("copyToHost(hostShapeBuf");
			shapeBuf->copyToHost(hostShapeBuf);
		}

		btAlignedObjectArray<btVector3> uniqueEdges;
		{
			BT_PROFILE("copyToHost(gpuUniqueEdges)");
			gpuUniqueEdges.copyToHost(uniqueEdges);
		}

		btAlignedObjectArray<int> indices;
		{
			BT_PROFILE("copyToHost(gpuIndices)");
			gpuIndices.copyToHost(indices);
		}

		
		{
			BT_PROFILE("copyToHost(convexData)");
			convexData.copyToHost(hostConvexData);
		}

		btAssert(m_hostPairs.size() == nPairs);
		m_hostContactOut.reserve(nPairs);
		
		
		if (findSeparatingAxisOnGpu && sepNormals.size())
		{
			sepNormals.copyToHost(hostNormals);
			hasSeparatingNormals.copyToHost(hostHasSep);
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

		
			

			bool validateFindSeparatingAxis = false;//true;
			if (validateFindSeparatingAxis)
			{
			
				btVector3 sepNormalWorldSpace;
				bool foundSepAxis =false;

				BT_PROFILE("findSeparatingAxis");
				foundSepAxis = findSeparatingAxis(
							hostConvexData.at(shapeA), 
							hostConvexData.at(shapeB),
							hostBodyBuf[indexA].m_pos,
							hostBodyBuf[indexA].m_quat,
							hostBodyBuf[indexB].m_pos,
							hostBodyBuf[indexB].m_quat,

							vertices,uniqueEdges,faces,indices,sepNormalWorldSpace);
				if ((int)foundSepAxis != hostHasSep[i])
				{
					printf("not matching boolean with gpu at %d\n",i);
				} 
				if (foundSepAxis && (btFabs(sepNormalWorldSpace[0]-hostNormals[i].x)>1e06f))
				{
					printf("not matching normal %f != %f with gpu at %d\n",sepNormalWorldSpace[0],hostNormals[i].x,i);
				}
			}

			float4 contactsOut[MAX_VERTS];
			int contactCapacity=MAX_VERTS;
			int numContactsOut=0;
			
			if (hostHasSep[i])
			{
				BT_PROFILE("hostHasSep");
		
				btScalar minDist = -1;
				btScalar maxDist = 0.1;

				
				

				btTransform trA,trB;
				{
				//BT_PROFILE("transform computation");
				//trA.setIdentity();
				trA.setOrigin(btVector3(hostBodyBuf[indexA].m_pos.x,hostBodyBuf[indexA].m_pos.y,hostBodyBuf[indexA].m_pos.z));
				trA.setRotation(btQuaternion(hostBodyBuf[indexA].m_quat.x,hostBodyBuf[indexA].m_quat.y,hostBodyBuf[indexA].m_quat.z,hostBodyBuf[indexA].m_quat.w));
				
				//trB.setIdentity();
				trB.setOrigin(btVector3(hostBodyBuf[indexB].m_pos.x,hostBodyBuf[indexB].m_pos.y,hostBodyBuf[indexB].m_pos.z));
				trB.setRotation(btQuaternion(hostBodyBuf[indexB].m_quat.x,hostBodyBuf[indexB].m_quat.y,hostBodyBuf[indexB].m_quat.z,hostBodyBuf[indexB].m_quat.w));
				}

		
				float4 worldVertsB1[MAX_VERTS];
				float4 worldVertsB2[MAX_VERTS];
				int capacityWorldVerts = MAX_VERTS;

                btQuaternion trAorn = trA.getRotation();
                btQuaternion trBorn = trB.getRotation();
                

				numContactsOut = clipHullAgainstHull(hostNormals[i], 
					hostConvexData.at(shapeA), 
					hostConvexData.at(shapeB),
								(float4&)trA.getOrigin(), (Quaternion&)trAorn,
								(float4&)trB.getOrigin(), (Quaternion&)trBorn,
								worldVertsB1,worldVertsB2,capacityWorldVerts,
								minDist, maxDist,(float4*)&vertices[0],&faces[0],&indices[0],contactsOut,contactCapacity);
				
			}
			if (numContactsOut>0)
			{
				float4 normalOnSurfaceB = -(float4&)hostNormals[i];
                
				if (reductionOnGpu)
				{
					btOpenCLArray<int> contactCount(m_context, m_queue);
					contactCount.push_back(numContactsOut);
					btOpenCLArray<int> totalContactsOut(m_context, m_queue);
					totalContactsOut.push_back(nContacts);

					

					btOpenCLArray<int> contactOffsets(m_context, m_queue);
					contactOffsets.push_back(0);
					
					btOpenCLArray<float4> closestPointOnBWorld(m_context,m_queue);
					
					closestPointOnBWorld.resize(numContactsOut,false);
					closestPointOnBWorld.copyFromHostPointer(contactsOut,numContactsOut,0,true);
					//closestPointOnBWorld.copyFromHost(resultOut.m_closestPointInBs);

					
					btOpenCLArray<float4> normalOnSurface(m_context,m_queue);
					normalOnSurface.push_back(normalOnSurfaceB);

					BT_PROFILE("extractManifoldAndAddContactKernel");
					btBufferInfoCL bInfo[] = { 
						btBufferInfoCL( pairs->getBufferCL(), true ), 
						btBufferInfoCL( bodyBuf->getBufferCL(),true), 
						btBufferInfoCL( closestPointOnBWorld.getBufferCL(),true),
						btBufferInfoCL( normalOnSurface.getBufferCL(),true),
						btBufferInfoCL( contactCount.getBufferCL(),true),
						btBufferInfoCL( contactOffsets.getBufferCL(),true),
						btBufferInfoCL( contactOut->getBufferCL()),
						btBufferInfoCL( totalContactsOut.getBufferCL())
					};

					btLauncherCL launcher(m_queue, m_extractManifoldAndAddContactKernel);
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
					int num = 1;//nPairs;
					
					launcher.setConst( num);//nPairs  );
					launcher.setConst( i);//nPairs  );
					//int contactCapacity = MAX_BROADPHASE_COLLISION_CL;
					//launcher.setConst(contactCapacity);
					
					launcher.launch1D( num);
					clFinish(m_queue);

					nContacts = totalContactsOut.at(0);
			
				} else
				{
					BT_PROFILE("overlap");
					float4 centerOut;
					int contactIdx[4]={-1,-1,-1,-1};

					int numPoints = 0;
					
					{
						BT_PROFILE("extractManifold");
						numPoints = extractManifold(contactsOut, numContactsOut, normalOnSurfaceB, centerOut,  contactIdx);
					}
					
					btAssert(numPoints);
					
					Contact4& contact = m_hostContactOut[nContacts];
					contact.m_batchIdx = i;
					contact.m_bodyAPtrAndSignBit = (hostBodyBuf[indexA].m_invMass==0)? -m_hostPairs[i].x:m_hostPairs[i].x;
					contact.m_bodyBPtrAndSignBit = (hostBodyBuf[indexB].m_invMass==0)? -m_hostPairs[i].y:m_hostPairs[i].y;

					contact.m_frictionCoeffCmp = 45874;
					contact.m_restituitionCoeffCmp = 0;
					
					float distance = 0.f;
					for (int p=0;p<numPoints;p++)
					{
						contact.m_worldPos[p] = contactsOut[contactIdx[p]];
						contact.m_worldNormal = normalOnSurfaceB; 
					}
					contact.m_worldNormal.w = numPoints;
					nContacts++;
				} 
			}
		}

		
		m_hostContactOut.resize(nContacts);

		if (!reductionOnGpu)
		{
			BT_PROFILE("copyFromHost(m_hostContactOut");
			contactOut->copyFromHost(m_hostContactOut);
		}
	}
}
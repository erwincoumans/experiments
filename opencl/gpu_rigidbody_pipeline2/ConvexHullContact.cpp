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

//#define BT_DEBUG_SAT_FACE

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
#include "LinearMath/btAabbUtil2.h"


GpuSatCollision::GpuSatCollision(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),
m_findSeparatingAxisKernel(0),
m_totalContactsOut(m_context, m_queue)
{
	m_totalContactsOut.push_back(0);
	
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

        m_findClippingFacesKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "findClippingFacesKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);

        m_clipFacesAndContactReductionKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipFacesAndContactReductionKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);        

		m_clipHullHullConcaveConvexKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipHullHullConcaveConvexKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);

		m_extractManifoldAndAddContactKernel = btOpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "extractManifoldAndAddContactKernel",&errNum,satClipContactsProg);
		btAssert(errNum==CL_SUCCESS);

	} else
	{
		m_clipHullHullKernel=0;
        m_findClippingFacesKernel = 0;
        m_clipFacesAndContactReductionKernel = 0;
		m_clipHullHullConcaveConvexKernel = 0;
		m_extractManifoldAndAddContactKernel = 0;
	}
	

}

GpuSatCollision::~GpuSatCollision()
{
	if (m_findSeparatingAxisKernel)
		clReleaseKernel(m_findSeparatingAxisKernel);

    
    if (m_findClippingFacesKernel)
        clReleaseKernel(m_findClippingFacesKernel);
   
    if (m_clipFacesAndContactReductionKernel)
        clReleaseKernel(m_clipFacesAndContactReductionKernel);
    
	if (m_clipHullHullKernel)
		clReleaseKernel(m_clipHullHullKernel);
	if (m_clipHullHullConcaveConvexKernel)
		clReleaseKernel(m_clipHullHullConcaveConvexKernel);
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
		btVector3 vertex = vertices[hull.m_vertexOffset+i];
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
	const float4& sep_axis, const btAlignedObjectArray<btVector3>& verticesA,const btAlignedObjectArray<btVector3>& verticesB,btScalar& depth)
{
	btScalar Min0,Max0;
	btScalar Min1,Max1;
	project(hullA,posA,ornA,sep_axis,verticesA, Min0, Max0);
	project(hullB,posB,ornB, sep_axis,verticesB, Min1, Max1);

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
	const btAlignedObjectArray<btVector3>& verticesA, 
	const btAlignedObjectArray<btVector3>& uniqueEdgesA, 
	const btAlignedObjectArray<btGpuFace>& facesA,
	const btAlignedObjectArray<int>& indicesA,
	const btAlignedObjectArray<btVector3>& verticesB, 
	const btAlignedObjectArray<btVector3>& uniqueEdgesB, 
	const btAlignedObjectArray<btGpuFace>& facesB,
	const btAlignedObjectArray<int>& indicesB,

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
	const float4 deltaC2 = c0 - c1;
//#endif

	btScalar dmin = FLT_MAX;
	int curPlaneTests=0;

	int numFacesA = hullA.m_numFaces;
	// Test normals from hullA
	for(int i=0;i<numFacesA;i++)
	{
		const float4& normal = (float4&)facesA[hullA.m_faceOffset+i].m_plane;
		float4 faceANormalWS = qtRotate(ornA,normal);

		if (dot3F4(deltaC2,faceANormalWS)<0)
			faceANormalWS*=-1.f;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB, DeltaC2, faceANormalWS, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		
		btScalar d;
		if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,faceANormalWS, verticesA, verticesB,d))
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
		float4 normal = (float4&)facesB[hullB.m_faceOffset+i].m_plane;
		float4 WorldNormal = qtRotate(ornB, normal);

		if (dot3F4(deltaC2,WorldNormal)<0)
		{
			WorldNormal*=-1.f;
		}
		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, WorldNormal, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		if(!TestSepAxis(hullA, hullB,posA,ornA,posB,ornB,WorldNormal,verticesA,verticesB,d))
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
		const float4& edge0 = (float4&) uniqueEdgesA[hullA.m_uniqueEdgesOffset+e0];
		float4 edge0World = qtRotate(ornA,(float4&)edge0);

		for(int e1=0;e1<hullB.m_numUniqueEdges;e1++)
		{
			const btVector3 edge1 = uniqueEdgesB[hullB.m_uniqueEdgesOffset+e1];
			float4 edge1World = qtRotate(ornB,(float4&)edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero((btVector3&)crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(deltaC2,crossje)<0)
					crossje*=-1.f;


#ifdef TEST_INTERNAL_OBJECTS
				gExpectedNbTests++;
				if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, Cross, hullA, hullB, dmin))
					continue;
				gActualNbTests++;
#endif

				btScalar dist;
				if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,crossje, verticesA,verticesB,dist))
					return false;

				if(dist<dmin)
				{
					dmin = dist;
					sep = (btVector3&)crossje;
				}
			}
		}

	}

	
	if((dot3F4(-deltaC2,(float4&)sep))>0.0f)
		sep = -sep;

	return true;
}



int clipFaceAgainstHull(const float4& separatingNormal, const ConvexPolyhedronCL* hullA,  
	const float4& posA, const Quaternion& ornA, float4* worldVertsB1, int numWorldVertsB1,
	float4* worldVertsB2, int capacityWorldVertsB2,
	const float minDist, float maxDist,
	const float4* verticesA,	const btGpuFace* facesA,	const int* indicesA,
	//const float4* verticesB,	const btGpuFace* facesB,	const int* indicesB,
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
				facesA[hullA->m_faceOffset+face].m_plane.x, 
				facesA[hullA->m_faceOffset+face].m_plane.y, 
				facesA[hullA->m_faceOffset+face].m_plane.z,0.f);
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

	btGpuFace polyA = facesA[hullA->m_faceOffset+closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numContacts = numWorldVertsB1;
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const float4 a = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+e0]];
		const float4 b = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
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
				//printf("depth=%f\n",depth);
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
	const float4* verticesA,	const btGpuFace* facesA,	const int* indicesA,
	const float4* verticesB,	const btGpuFace* facesB,	const int* indicesB,

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
		if (hullB.m_numFaces!=1)
		{
			//printf("wtf\n");
		}
		static bool once = true;
		//printf("separatingNormal=%f,%f,%f\n",separatingNormal.x,separatingNormal.y,separatingNormal.z);
		
		for(int face=0;face<hullB.m_numFaces;face++)
		{
#ifdef BT_DEBUG_SAT_FACE
			if (once)
				printf("face %d\n",face);
			const btGpuFace* faceB = &facesB[hullB.m_faceOffset+face];
			if (once)
			{
				for (int i=0;i<faceB->m_numIndices;i++)
				{
					float4 vert = verticesB[hullB.m_vertexOffset+indicesB[faceB->m_indexOffset+i]];
					printf("vert[%d] = %f,%f,%f\n",i,vert.x,vert.y,vert.z);
				}
			}
#endif //BT_DEBUG_SAT_FACE
			//if (facesB[hullB.m_faceOffset+face].m_numIndices>2)
			{
				const float4 Normal = make_float4(facesB[hullB.m_faceOffset+face].m_plane.x, 
					facesB[hullB.m_faceOffset+face].m_plane.y, facesB[hullB.m_faceOffset+face].m_plane.z,0.f);
				const float4 WorldNormal = qtRotate(ornB, Normal);
#ifdef BT_DEBUG_SAT_FACE
				if (once)
					printf("faceNormal = %f,%f,%f\n",Normal.x,Normal.y,Normal.z);
#endif
				float d = dot3F4(WorldNormal,separatingNormal);
				if (d > dmax)
				{
					dmax = d;
					closestFaceB = face;
				}
			}
		}
		once = false;
	}

	
	btAssert(closestFaceB>=0);
	{
		//BT_PROFILE("worldVertsB1");
		const btGpuFace& polyB = facesB[hullB.m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4& b = verticesB[hullB.m_vertexOffset+indicesB[polyB.m_indexOffset+e0]];
			worldVertsB1[numWorldVertsB1++] = transform(b,posB,ornB);
		}
	}

	if (closestFaceB>=0)
	{
		//BT_PROFILE("clipFaceAgainstHull");
		numContactsOut = clipFaceAgainstHull((float4&)separatingNormal, &hullA, 
				posA,ornA,
				worldVertsB1,numWorldVertsB1,worldVertsB2,capacityWorldVerts, minDist, maxDist,
				verticesA,				facesA,				indicesA,
				contactsOut,contactCapacity);
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

void GpuSatCollision::computeConcaveConvexContactsGPUSATSingle(
	int bodyIndexA, int bodyIndexB,
			int collidableIndexA, int collidableIndexB,

			const btAlignedObjectArray<RigidBodyBase::Body>* bodyBuf, 
			const btAlignedObjectArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, 
			int& nContacts, const ChNarrowphase::Config& cfg , 
			const btAlignedObjectArray<ConvexPolyhedronCL>& hostConvexDataB,
			const btAlignedObjectArray<btVector3>& verticesB,
			const btAlignedObjectArray<btVector3>& uniqueEdgesB,
			const btAlignedObjectArray<btGpuFace>& facesB,
			const btAlignedObjectArray<int>& indicesB,
			const btAlignedObjectArray<btCollidable>& hostCollidablesB,
			btAlignedObjectArray<btYetAnotherAabb>& hostAabbs, 
			int numObjects,
			int maxTriConvexPairCapacity,
			btAlignedObjectArray<int4>& triangleConvexPairs,
			int& numTriConvexPairsOut)
{

	int shapeA = hostCollidablesB[collidableIndexA].m_shapeIndex;
	int shapeB = hostCollidablesB[collidableIndexB].m_shapeIndex;

	int numFaces = hostConvexDataB[shapeA].m_numFaces;
							
	int numActualConcaveConvexTests = 0;

	for (int f=0;f<numFaces;f++)
								
	{
		BT_PROFILE("each face");	
		const btGpuFace& face = facesB[hostConvexDataB[shapeA].m_faceOffset+f];


		//for now we ignore quads, only tris are allowed
		btAssert(face.m_numIndices==3);
                                    
		if (face.m_numIndices==3)
		{
			btAlignedObjectArray<btCollidable> hostCollidablesA;
			btAlignedObjectArray<ConvexPolyhedronCL> hostConvexDataA;
			int collidableIndexA = 0;
			btAlignedObjectArray<btVector3> uniqueEdgesA;
			btAlignedObjectArray<btGpuFace> facesA;
			btAlignedObjectArray<int> indicesA;
                                        
			btAlignedObjectArray<btVector3> verticesA;
                                        
                                        
                                        
                                        
			btCollidable colA;
			colA.m_shapeIndex = 0;
			colA.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
			hostCollidablesA.push_back(colA);
                                        
			ConvexPolyhedronCL convexPolyhedronA;

			//add 3 vertices of the triangle
			convexPolyhedronA.m_numVertices = 3;
			convexPolyhedronA.m_vertexOffset = 0;
			btVector3 localCenter(0,0,0);
			btVector3 triMinAabb, triMaxAabb;
			triMinAabb.setValue(1e30,1e30,1e30);
			triMaxAabb.setValue(-1e30,-1e30,-1e30);
										
			{
				BT_PROFILE("extract triangle verts");
				for (int i=0;i<3;i++)
				{
					int index = indicesB[face.m_indexOffset+i];
					btVector3 vert = verticesB[hostConvexDataB[shapeA].m_vertexOffset+index];
					verticesA.push_back(vert);
					triMinAabb.setMin(vert);
					triMaxAabb.setMax(vert);
					localCenter+=vert;
				}
			}
                                        
			//check for AABB overlap first
			btVector3 convexMinAabb(hostAabbs[bodyIndexB].m_min[0],hostAabbs[bodyIndexB].m_min[1],hostAabbs[bodyIndexB].m_min[2]);
			btVector3 convexMaxAabb(hostAabbs[bodyIndexB].m_max[0],hostAabbs[bodyIndexB].m_max[1],hostAabbs[bodyIndexB].m_max[2]);

			bool overlapAabbTriConvex=false;
			{
				BT_PROFILE("TestAabbAgainstAabb2");
				overlapAabbTriConvex = TestAabbAgainstAabb2(triMinAabb,triMaxAabb,convexMinAabb,convexMaxAabb);
			}

			if (!overlapAabbTriConvex)
				continue;
										
			if (1)
			{
				BT_PROFILE("concave-convex actual test");
				int localCC=0;
				numActualConcaveConvexTests++;

				//a triangle has 3 unique edges
				convexPolyhedronA.m_numUniqueEdges = 3;
				convexPolyhedronA.m_uniqueEdgesOffset = 0;
				uniqueEdgesA.push_back(verticesA[1]-verticesA[0]);
				uniqueEdgesA.push_back(verticesA[2]-verticesA[1]);
				uniqueEdgesA.push_back(verticesA[0]-verticesA[2]);

				convexPolyhedronA.m_faceOffset = 0;
                                        
				btVector3 normal(face.m_plane.x,face.m_plane.y,face.m_plane.z);
                                        
				//front size of triangle
				{
					btGpuFace gpuFace;
					gpuFace.m_indexOffset=indicesA.size();
					indicesA.push_back(0);
					indicesA.push_back(1);
					indicesA.push_back(2);
					btScalar c = face.m_plane.w;
					gpuFace.m_plane.x = normal[0];
					gpuFace.m_plane.y = normal[1];
					gpuFace.m_plane.z = normal[2];
					gpuFace.m_plane.w = c;
					gpuFace.m_numIndices=3;
					facesA.push_back(gpuFace);
				}

				//back size of triangle
	#if 1
				{
					btGpuFace gpuFace;
					gpuFace.m_indexOffset=indicesA.size();
					indicesA.push_back(2);
					indicesA.push_back(1);
					indicesA.push_back(0);
					btScalar c = (normal.dot(verticesA[0]));
					btScalar c1 = -face.m_plane.w;
					btAssert(c==c1);
					gpuFace.m_plane.x = -normal[0];
					gpuFace.m_plane.y = -normal[1];
					gpuFace.m_plane.z = -normal[2];
					gpuFace.m_plane.w = c;
					gpuFace.m_numIndices=3;
					facesA.push_back(gpuFace);
				}

				bool addEdgePlanes = true;
				if (addEdgePlanes)
				{
					int numVertices=3;
					int prevVertex = numVertices-1;
					for (int i=0;i<numVertices;i++)
					{
						btGpuFace gpuFace;
                                                
						btVector3 v0 = verticesA[i];
						btVector3 v1 = verticesA[prevVertex];
                                                
						btVector3 edgeNormal = (normal.cross(v1-v0)).normalize();
						btScalar c = -edgeNormal.dot(v0);

						gpuFace.m_numIndices = 2;
						gpuFace.m_indexOffset=indicesA.size();
						indicesA.push_back(i);
						indicesA.push_back(prevVertex);
                                                
						gpuFace.m_plane.x = edgeNormal[0];
						gpuFace.m_plane.y = edgeNormal[1];
						gpuFace.m_plane.z = edgeNormal[2];
						gpuFace.m_plane.w = c;
						facesA.push_back(gpuFace);
						prevVertex = i;
					}
				}
	#endif

				convexPolyhedronA.m_numFaces = facesA.size();
				convexPolyhedronA.m_localCenter = localCenter*(1./3.);


                                        
				hostConvexDataA.push_back(convexPolyhedronA);
                                        
				{
					BT_PROFILE("computeConvexConvexContactsGPUSATSingle");


				/*  m_internalData->m_gpuSatCollision->computeConvexConvexContactsGPUSATSingle(
																							bodyIndexA, bodyIndexB,
																							collidableIndexA, collidableIndexB,
																							m_internalData->m_bodyBufferCPU,
																							0,
																							m_internalData->m_pBufContactOutGPU,
																							nContactOut,cfgNP,
																							hostConvexDataA,
																							convexPolyhedraB,
																							verticesA,
																							uniqueEdgesA,
																							facesA,
																							indicesA,
																							verticesB,
																							uniqueEdgesB,
																							facesB,
																							indicesB,
																							hostCollidablesA,
																							hostCollidablesB);
																							*/


				computeConvexConvexContactsGPUSATSingle(
																							bodyIndexB,bodyIndexA,
																							collidableIndexB,collidableIndexA,
																							bodyBuf,
																							0,
																							contactOut,
																							nContacts,cfg,
																							hostConvexDataB,
																							hostConvexDataA,
																							verticesB,
																							uniqueEdgesB,
																							facesB,
																							indicesB,
																								verticesA,
																							uniqueEdgesA,
																							facesA,
																							indicesA,
																							hostCollidablesB,
																							hostCollidablesA);
										
				}
			}

		}
	}
}


void GpuSatCollision::clipHullHullSingle(
			int bodyIndexA, int bodyIndexB,
			int collidableIndexA, int collidableIndexB,

			const btAlignedObjectArray<RigidBodyBase::Body>* bodyBuf, 
			const btAlignedObjectArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, 
			int& nContacts, const ChNarrowphase::Config& cfg , 
			
			const btAlignedObjectArray<ConvexPolyhedronCL>& hostConvexDataA,
			const btAlignedObjectArray<ConvexPolyhedronCL>& hostConvexDataB,
	
			const btAlignedObjectArray<btVector3>& verticesA, 
			const btAlignedObjectArray<btVector3>& uniqueEdgesA, 
			const btAlignedObjectArray<btGpuFace>& facesA,
			const btAlignedObjectArray<int>& indicesA,
	
			const btAlignedObjectArray<btVector3>& verticesB,
			const btAlignedObjectArray<btVector3>& uniqueEdgesB,
			const btAlignedObjectArray<btGpuFace>& facesB,
			const btAlignedObjectArray<int>& indicesB,

			const btAlignedObjectArray<btCollidable>& hostCollidablesA,
			const btAlignedObjectArray<btCollidable>& hostCollidablesB,
			const btVector3& sepNormalWorldSpace,int numContactsOut
			)
{

	ConvexPolyhedronCL hullA, hullB;
    
    btCollidable colA = hostCollidablesA[collidableIndexA];
    hullA = hostConvexDataA[colA.m_shapeIndex];
    //printf("numvertsA = %d\n",hullA.m_numVertices);
    
    
    btCollidable colB = hostCollidablesB[collidableIndexB];
    hullB = hostConvexDataB[colB.m_shapeIndex];
    //printf("numvertsB = %d\n",hullB.m_numVertices);
    
	
	float4 contactsOut[MAX_VERTS];
	int contactCapacity = MAX_VERTS;

#ifdef _WIN32
	btAssert(_finite(bodyBuf->at(bodyIndexA).m_pos.x));
	btAssert(_finite(bodyBuf->at(bodyIndexB).m_pos.x));
#endif
	
	
	{
		
		float4 worldVertsB1[MAX_VERTS];
		float4 worldVertsB2[MAX_VERTS];
		int capacityWorldVerts = MAX_VERTS;

		float4 hostNormal = make_float4(sepNormalWorldSpace.getX(),sepNormalWorldSpace.getY(),sepNormalWorldSpace.getZ(),0.f);
		int shapeA = hostCollidablesA[collidableIndexA].m_shapeIndex;
		int shapeB = hostCollidablesB[collidableIndexB].m_shapeIndex;

		btScalar minDist = -1;
		btScalar maxDist = 0.1;

		        

		btTransform trA,trB;
		{
		//BT_PROFILE("transform computation");
		//trA.setIdentity();
		trA.setOrigin(btVector3(bodyBuf->at(bodyIndexA).m_pos.x,bodyBuf->at(bodyIndexA).m_pos.y,bodyBuf->at(bodyIndexA).m_pos.z));
		trA.setRotation(btQuaternion(bodyBuf->at(bodyIndexA).m_quat.x,bodyBuf->at(bodyIndexA).m_quat.y,bodyBuf->at(bodyIndexA).m_quat.z,bodyBuf->at(bodyIndexA).m_quat.w));
				
		//trB.setIdentity();
		trB.setOrigin(btVector3(bodyBuf->at(bodyIndexB).m_pos.x,bodyBuf->at(bodyIndexB).m_pos.y,bodyBuf->at(bodyIndexB).m_pos.z));
		trB.setRotation(btQuaternion(bodyBuf->at(bodyIndexB).m_quat.x,bodyBuf->at(bodyIndexB).m_quat.y,bodyBuf->at(bodyIndexB).m_quat.z,bodyBuf->at(bodyIndexB).m_quat.w));
		}

		btQuaternion trAorn = trA.getRotation();
        btQuaternion trBorn = trB.getRotation();
        
		int numContactsOut = clipHullAgainstHull(hostNormal, 
						hostConvexDataA.at(shapeA), 
						hostConvexDataB.at(shapeB),
								(float4&)trA.getOrigin(), (Quaternion&)trAorn,
								(float4&)trB.getOrigin(), (Quaternion&)trBorn,
								worldVertsB1,worldVertsB2,capacityWorldVerts,
								minDist, maxDist,
								(float4*)&verticesA[0],	&facesA[0],&indicesA[0],
								(float4*)&verticesB[0],	&facesB[0],&indicesB[0],
								
								contactsOut,contactCapacity);

		if (numContactsOut>0)
		{
			BT_PROFILE("overlap");

			float4 normalOnSurfaceB = -(float4&)hostNormal;
			float4 centerOut;
			int contactIdx[4]={-1,-1,-1,-1};

			int numPoints = 0;
					
			{
				BT_PROFILE("extractManifold");
				numPoints = extractManifold(contactsOut, numContactsOut, normalOnSurfaceB, centerOut,  contactIdx);
			}
					
			btAssert(numPoints);
					
			if (contactOut->size())
			{
				BT_PROFILE("copyToHost(m_hostContactOut");
				contactOut->copyToHost(m_hostContactOut);
			} else
			{
				m_hostContactOut.resize(0);
			}
			m_hostContactOut.expand();
			Contact4& contact = m_hostContactOut[nContacts];
			contact.m_batchIdx = 0;//i;
			contact.m_bodyAPtrAndSignBit = (bodyBuf->at(bodyIndexA).m_invMass==0)? -bodyIndexA:bodyIndexA;
			contact.m_bodyBPtrAndSignBit = (bodyBuf->at(bodyIndexB).m_invMass==0)? -bodyIndexB:bodyIndexB;

			contact.m_frictionCoeffCmp = 45874;
			contact.m_restituitionCoeffCmp = 0;
					
			float distance = 0.f;
			for (int p=0;p<numPoints;p++)
			{
				contact.m_worldPos[p] = contactsOut[contactIdx[p]];
				contact.m_worldNormal = normalOnSurfaceB; 
			}
			contact.m_worldNormal.w = numPoints;
			
			{
				BT_PROFILE("contactOut->copyFromHost");
				contactOut->copyFromHost(m_hostContactOut);
			}
			nContacts++;
		}
	}
}


void GpuSatCollision::computeConvexConvexContactsGPUSATSingle(
			int bodyIndexA, int bodyIndexB,
			int collidableIndexA, int collidableIndexB,

			const btAlignedObjectArray<RigidBodyBase::Body>* bodyBuf, 
			const btAlignedObjectArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, 
			int& nContacts, const ChNarrowphase::Config& cfg , 
			
			const btAlignedObjectArray<ConvexPolyhedronCL>& hostConvexDataA,
			const btAlignedObjectArray<ConvexPolyhedronCL>& hostConvexDataB,
	
			const btAlignedObjectArray<btVector3>& verticesA, 
			const btAlignedObjectArray<btVector3>& uniqueEdgesA, 
			const btAlignedObjectArray<btGpuFace>& facesA,
			const btAlignedObjectArray<int>& indicesA,
	
			const btAlignedObjectArray<btVector3>& verticesB,
			const btAlignedObjectArray<btVector3>& uniqueEdgesB,
			const btAlignedObjectArray<btGpuFace>& facesB,
			const btAlignedObjectArray<int>& indicesB,

			const btAlignedObjectArray<btCollidable>& hostCollidablesA,
			const btAlignedObjectArray<btCollidable>& hostCollidablesB
			)
{

	ConvexPolyhedronCL hullA, hullB;
    
	btVector3 sepNormalWorldSpace;

	

    btCollidable colA = hostCollidablesA[collidableIndexA];
    hullA = hostConvexDataA[colA.m_shapeIndex];
    //printf("numvertsA = %d\n",hullA.m_numVertices);
    
    
    btCollidable colB = hostCollidablesB[collidableIndexB];
    hullB = hostConvexDataB[colB.m_shapeIndex];
    //printf("numvertsB = %d\n",hullB.m_numVertices);
    
	
	float4 contactsOut[MAX_VERTS];
	int contactCapacity = MAX_VERTS;
	int numContactsOut=0;


#ifdef _WIN32
	btAssert(_finite(bodyBuf->at(bodyIndexA).m_pos.x));
	btAssert(_finite(bodyBuf->at(bodyIndexB).m_pos.x));
#endif
	
		bool foundSepAxis = findSeparatingAxis(hullA,hullB,
							bodyBuf->at(bodyIndexA).m_pos,
							bodyBuf->at(bodyIndexA).m_quat,
							bodyBuf->at(bodyIndexB).m_pos,
							bodyBuf->at(bodyIndexB).m_quat,

							verticesA,uniqueEdgesA,facesA,indicesA,
							verticesB,uniqueEdgesB,facesB,indicesB,
							
							sepNormalWorldSpace
							);

	
	if (foundSepAxis)
	{
		clipHullHullSingle(
			bodyIndexA, bodyIndexB,
			collidableIndexA, collidableIndexB,
			bodyBuf, 
			shapeBuf,
			contactOut, 
			nContacts, cfg , 
			
			hostConvexDataA,
			hostConvexDataB,
	
			verticesA, 
			uniqueEdgesA, 
			facesA,
			indicesA,
	
			verticesB,
			uniqueEdgesB,
			facesB,
			indicesB,

			hostCollidablesA,
			hostCollidablesB,
			sepNormalWorldSpace,numContactsOut);

	}


}


void GpuSatCollision::computeConvexConvexContactsGPUSAT_sequential( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const ChNarrowphase::Config& cfg , 
			const btOpenCLArray<ConvexPolyhedronCL>& convexData,
			const btOpenCLArray<btVector3>& gpuVertices,
			const btOpenCLArray<btVector3>& gpuUniqueEdges,
			const btOpenCLArray<btGpuFace>& gpuFaces,
			const btOpenCLArray<int>& gpuIndices,
			const btOpenCLArray<btCollidable>& gpuCollidables,
			const btOpenCLArray<btYetAnotherAabb>& clAabbs, 
			int numObjects,
			int maxTriConvexPairCapacity,
			btOpenCLArray<int4>& triangleConvexPairs,
			int& numTriConvexPairsOut
			)
{
	if (!nPairs)
		return;

	btAlignedObjectArray<btYetAnotherAabb> hostAabbs;
	clAabbs.copyToHost(hostAabbs);

	pairs->copyToHost(m_hostPairs);

	btAlignedObjectArray<RigidBodyBase::Body> hostBodyBuf;
	bodyBuf->copyToHost(hostBodyBuf);

	
	btAlignedObjectArray<ChNarrowphase::ShapeData> hostShapeBuf;
	shapeBuf->copyToHost(hostShapeBuf);

	btAlignedObjectArray<ConvexPolyhedronCL> hostConvexData;
	convexData.copyToHost(hostConvexData);

	btAlignedObjectArray<btVector3> hostVertices;
	gpuVertices.copyToHost(hostVertices);

	btAlignedObjectArray<btVector3> hostUniqueEdges;
	gpuUniqueEdges.copyToHost(hostUniqueEdges);
	btAlignedObjectArray<btGpuFace> hostFaces;
	gpuFaces.copyToHost(hostFaces);
	btAlignedObjectArray<int> hostIndices;
	gpuIndices.copyToHost(hostIndices);
	btAlignedObjectArray<btCollidable> hostCollidables;
	gpuCollidables.copyToHost(hostCollidables);

	btAlignedObjectArray<int4> hostTriangleConvexPairs;

	for (int i=0;i<nPairs;i++)
	{
		int bodyIndexA = m_hostPairs[i].x;
		int bodyIndexB = m_hostPairs[i].y;
		int collidableIndexA = hostBodyBuf[bodyIndexA].m_collidableIdx;
		int collidableIndexB = hostBodyBuf[bodyIndexB].m_collidableIdx;

		if (hostCollidables[collidableIndexA].m_shapeType == CollisionShape::SHAPE_CONCAVE_TRIMESH)
		{
			computeConcaveConvexContactsGPUSATSingle(bodyIndexA,bodyIndexB,
				collidableIndexA,collidableIndexB,&hostBodyBuf,&hostShapeBuf,contactOut,nContacts,cfg,
				hostConvexData, hostVertices,hostUniqueEdges,hostFaces,hostIndices,hostCollidables,
				hostAabbs,
				numObjects,
				maxTriConvexPairCapacity,
				hostTriangleConvexPairs,
				numTriConvexPairsOut);
				
			continue;
		}

		if (hostCollidables[collidableIndexB].m_shapeType == CollisionShape::SHAPE_CONCAVE_TRIMESH)
		{
			btAssert(0);
		}

		if (hostCollidables[collidableIndexA].m_shapeType != CollisionShape::SHAPE_CONVEX_HULL)
			continue;

		if (hostCollidables[collidableIndexB].m_shapeType != CollisionShape::SHAPE_CONVEX_HULL)
			continue;

		if (hostConvexData[collidableIndexA].m_numFaces<6)
			{
				computeConvexConvexContactsGPUSATSingle(bodyIndexB, bodyIndexA,
				collidableIndexB,collidableIndexA,&hostBodyBuf,&hostShapeBuf,contactOut,nContacts,cfg,
				hostConvexData,hostConvexData,
				hostVertices,hostUniqueEdges,hostFaces,hostIndices,
				hostVertices,hostUniqueEdges,hostFaces,hostIndices,hostCollidables,hostCollidables);
			} else
			{
				computeConvexConvexContactsGPUSATSingle(bodyIndexA,bodyIndexB,
				collidableIndexA,collidableIndexB,&hostBodyBuf,&hostShapeBuf,contactOut,nContacts,cfg,
				hostConvexData,hostConvexData,
				hostVertices,hostUniqueEdges,hostFaces,hostIndices,
				hostVertices,hostUniqueEdges,hostFaces,hostIndices,hostCollidables,hostCollidables);
			}

				
		//computeConvexConvexContactsGPUSATSingle(bodyIndexA,bodyIndexB,
		//	collidableIndexA,collidableIndexB,&hostBodyBuf,&hostShapeBuf,contactOut,nContacts,cfg,
		//	hostConvexData,hostConvexData,
		//	hostVertices,hostUniqueEdges,hostFaces,hostIndices,
		//	hostVertices,hostUniqueEdges,hostFaces,hostIndices,hostCollidables,hostCollidables);

	

	}

	if (hostTriangleConvexPairs.size())
	{
		triangleConvexPairs.copyFromHost(hostTriangleConvexPairs);
	}
}


 void   clipHullHullKernel( btAlignedObjectArray<int2>& pairs,
                                   btAlignedObjectArray<struct BodyData>& rigidBodies,
                                   const btAlignedObjectArray<struct btCollidableGpu>&collidables,
                                   btAlignedObjectArray<struct ConvexPolyhedronCL>& convexShapes,
                                   btAlignedObjectArray<float4>& vertices,
                                   btAlignedObjectArray<float4>& uniqueEdges,
                                   btAlignedObjectArray<btGpuFace>& faces,
                                   btAlignedObjectArray<int>& indices,
                                   btAlignedObjectArray<float4>&separatingNormals,
                                   btAlignedObjectArray<int>& hasSeparatingAxis,
                                   btAlignedObjectArray<Contact4>&globalContactsOut,
                                   int* nGlobalContactsOut,
                                   int numPairs);

void   findClippingFacesKernel( btAlignedObjectArray<int2>& pairs,
                               btAlignedObjectArray< BodyData>& rigidBodies,
                               const btAlignedObjectArray< btCollidableGpu>&collidables,
                               btAlignedObjectArray< ConvexPolyhedronCL>& convexShapes,
                               btAlignedObjectArray<float4>& vertices,
                               btAlignedObjectArray<float4>& uniqueEdges,
                               btAlignedObjectArray<btGpuFace>& faces,
                               btAlignedObjectArray<int>& indices,
                               btAlignedObjectArray<float4>&separatingNormals,
                               btAlignedObjectArray<int>& hasSeparatingAxis,
                               btAlignedObjectArray<int4>&clippingFacesOut,
                               btAlignedObjectArray<float4>& worldVertsA1,
                               btAlignedObjectArray<float4>& worldNormalsA1,
                               btAlignedObjectArray<float4>& worldVertsB1,
                               int capacityWorldVerts,
                               int numPairs);
    


void   clipFacesAndContactReductionKernel( btAlignedObjectArray<int2>& pairs,
                                          btAlignedObjectArray< BodyData>& rigidBodies,
											btAlignedObjectArray<float4>&separatingNormals,
                                          btAlignedObjectArray<int>& hasSeparatingAxis,
                                          btAlignedObjectArray<Contact4>&globalContactsOut,
                                          btAlignedObjectArray<int4>& clippingFaces,
                                          btAlignedObjectArray<float4>& worldVertsA1,
                                          btAlignedObjectArray<float4>& worldNormalsA1,
                                          btAlignedObjectArray<float4>& worldVertsB1,
                                          btAlignedObjectArray<float4>& worldVertsB2,
                                          int* nGlobalContactsOut,
                                          int vertexFaceCapacity,
                                          int numPairs);

extern int g_globalId;



void GpuSatCollision::computeConvexConvexContactsGPUSAT( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ChNarrowphase::ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const ChNarrowphase::Config& cfg , 
			const btOpenCLArray<ConvexPolyhedronCL>& convexData,
			const btOpenCLArray<btVector3>& gpuVertices,
			const btOpenCLArray<btVector3>& gpuUniqueEdges,
			const btOpenCLArray<btGpuFace>& gpuFaces,
			const btOpenCLArray<int>& gpuIndices,
			const btOpenCLArray<btCollidable>& gpuCollidables,
			const btOpenCLArray<btYetAnotherAabb>& clAabbs, 
			int numObjects,
			int maxTriConvexPairCapacity,
			btOpenCLArray<int4>& triangleConvexPairsOut,
			int& numTriConvexPairsOut
			)
{
	if (!nPairs)
		return;

	BT_PROFILE("computeConvexConvexContactsGPUSAT");
   // printf("nContacts = %d\n",nContacts);
    
	btAlignedObjectArray<ConvexPolyhedronCL> hostConvexData;
	btAlignedObjectArray<RigidBodyBase::Body> hostBodyBuf;

	btAlignedObjectArray<float4> hostNormals;
	btAlignedObjectArray<int> hostHasSep;

	btOpenCLArray<float4> sepNormals(m_context,m_queue);
	sepNormals.resize(nPairs);
	btOpenCLArray<int> hasSeparatingNormals(m_context,m_queue);
	hasSeparatingNormals.resize(nPairs);
	
	int concaveCapacity=8192;
	btOpenCLArray<float4> concaveSepNormals(m_context,m_queue);
	concaveSepNormals.resize(concaveCapacity);

	btOpenCLArray<int> numConcavePairsOut(m_context,m_queue);
	numConcavePairsOut.push_back(0);


	bool findSeparatingAxisOnGpu = true;
	int numConcave =0;

	{
		clFinish(m_queue);
		if (findSeparatingAxisOnGpu)
		{
	
		
			BT_PROFILE("findSeparatingAxisKernel");
			btBufferInfoCL bInfo[] = { 
				btBufferInfoCL( pairs->getBufferCL(), true ), 
				btBufferInfoCL( bodyBuf->getBufferCL(),true), 
				btBufferInfoCL( gpuCollidables.getBufferCL(),true), 
				btBufferInfoCL( convexData.getBufferCL(),true),
				btBufferInfoCL( gpuVertices.getBufferCL(),true),
				btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				btBufferInfoCL( gpuFaces.getBufferCL(),true),
				btBufferInfoCL( gpuIndices.getBufferCL(),true),
				btBufferInfoCL( clAabbs.getBufferCL(),true),
				btBufferInfoCL( sepNormals.getBufferCL()),
				btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
				btBufferInfoCL( triangleConvexPairsOut.getBufferCL()),
				btBufferInfoCL( concaveSepNormals.getBufferCL()),
				btBufferInfoCL( numConcavePairsOut.getBufferCL())
			};
			btLauncherCL launcher(m_queue, m_findSeparatingAxisKernel);
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( nPairs  );
			launcher.setConst( maxTriConvexPairCapacity);
			int num = nPairs;
			launcher.launch1D( num);
			clFinish(m_queue);

			numConcave = numConcavePairsOut.at(0);
			if (numConcave > maxTriConvexPairCapacity)
				numConcave = maxTriConvexPairCapacity;

			triangleConvexPairsOut.resize(numConcave);

			//printf("numConcave  = %d\n",numConcave);
			
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


			btAlignedObjectArray<btVector3> verticesA;
			{
				BT_PROFILE("copyToHost(gpuVertices)");
				gpuVertices.copyToHost(verticesA);
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

			btAlignedObjectArray<btCollidable> cpuCollidables;
			gpuCollidables.copyToHost(cpuCollidables);


			for (int i=0;i<nPairs;i++)
			{
				int indexA = m_hostPairs[i].x;
				int indexB = m_hostPairs[i].y;
				
				


				int collidableA = hostBodyBuf[indexA].m_collidableIdx;
				int collidableB = hostBodyBuf[indexB].m_collidableIdx;

				if (cpuCollidables[collidableA].m_shapeType != CollisionShape::SHAPE_CONVEX_HULL)
					continue;

				if (cpuCollidables[collidableB].m_shapeType != CollisionShape::SHAPE_CONVEX_HULL)
					continue;

				int shapeA = cpuCollidables[collidableA].m_shapeIndex;
				int shapeB = cpuCollidables[collidableB].m_shapeIndex;
				

				//printf("hostBodyBuf[indexA].m_shapeType=%d\n",hostBodyBuf[indexA].m_shapeType);


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

							verticesA,uniqueEdges,faces,indices,
							verticesA,uniqueEdges,faces,indices,
							
							sepNormalWorldSpace);

				hostHasSep[i] = foundSepAxis;
				if (foundSepAxis)
					hostNormals[i] = make_float4(sepNormalWorldSpace.getX(),sepNormalWorldSpace.getY(),sepNormalWorldSpace.getZ(),0.f);
					{
					sepNormals.copyFromHost(hostNormals);
					hasSeparatingNormals.copyFromHost(hostHasSep);
				}
			}
		}

//		printf("hostNormals.size()=%d\n",hostNormals.size());
		//int numPairs = pairCount.at(0);
		
	}
#ifdef __APPLE__
 bool contactClippingOnGpu = true;
#else
 bool contactClippingOnGpu = true;
#endif
	
	if (contactClippingOnGpu)
	{
		//BT_PROFILE("clipHullHullKernel");

		
		m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);

		//concave-convex contact clipping

		if (0)//numConcave)
		{
			BT_PROFILE("clipHullHullConcaveConvexKernel");
			nContacts = m_totalContactsOut.at(0);
			btBufferInfoCL bInfo[] = { 
				btBufferInfoCL( triangleConvexPairsOut.getBufferCL(), true ), 
				btBufferInfoCL( bodyBuf->getBufferCL(),true), 
				btBufferInfoCL( gpuCollidables.getBufferCL(),true), 
				btBufferInfoCL( convexData.getBufferCL(),true),
				btBufferInfoCL( gpuVertices.getBufferCL(),true),
				btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				btBufferInfoCL( gpuFaces.getBufferCL(),true),
				btBufferInfoCL( gpuIndices.getBufferCL(),true),
				btBufferInfoCL( concaveSepNormals.getBufferCL()),
				btBufferInfoCL( contactOut->getBufferCL()),
				btBufferInfoCL( m_totalContactsOut.getBufferCL())	
			};
			btLauncherCL launcher(m_queue, m_clipHullHullConcaveConvexKernel);
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numConcave  );
			int num = numConcave;
			launcher.launch1D( num);
			clFinish(m_queue);
			nContacts = m_totalContactsOut.at(0);
		}


		//convex-convex contact clipping

        if (1)
		{
			BT_PROFILE("clipHullHullKernel");
#ifdef __APPLE__
#define BREAKUP_KERNEL
#endif

//#define DEBUG_CPU_CLIP
#ifdef DEBUG_CPU_CLIP
            
            btAlignedObjectArray<int2> hostPairs;
            pairs->copyToHost(hostPairs);
            btAlignedObjectArray<RigidBodyBase::Body> hostBodies;
            bodyBuf->copyToHost(hostBodies);
            btAlignedObjectArray<btCollidable> hostCollidables;
            gpuCollidables.copyToHost(hostCollidables);
            btAlignedObjectArray<ConvexPolyhedronCL> hostConvexShapeData;
            convexData.copyToHost(hostConvexShapeData);
            btAlignedObjectArray<btVector3> verticesA;
            gpuVertices.copyToHost(verticesA);
			btAlignedObjectArray<btGpuFace> faces;
            gpuFaces.copyToHost(faces);
			btAlignedObjectArray<btVector3> uniqueEdges;
            gpuUniqueEdges.copyToHost(uniqueEdges);
			btAlignedObjectArray<int> indices;
            gpuIndices.copyToHost(indices);
//            btAlignedObjectArray<float4> hostConcaveSepNormals;
  //          concaveSepNormals.copyToHost(hostConcaveSepNormals);
            btAlignedObjectArray<float4> hostSepNormals;
            sepNormals.copyToHost(hostSepNormals);
            btAlignedObjectArray<int> hostHasSepAxis;
            hasSeparatingNormals.copyToHost(hostHasSepAxis);
            btAlignedObjectArray<Contact4> hostContactOut;
            contactOut->copyToHost(hostContactOut);
            int gpuContactCapacity = contactOut->capacity();
            
            hostContactOut.resize(gpuContactCapacity);
            hostHasSepAxis.resize(gpuContactCapacity);
            
            int nGlobalContactsOut = 0;
            
            int prevGlobalContactOut = nGlobalContactsOut;
            
            int vertexFaceCapacity = 64;

            btAlignedObjectArray<float4> worldVertsA1;
            worldVertsA1.resizeNoInitialize(vertexFaceCapacity*nPairs);

            btAlignedObjectArray<float4> worldNormalsA1;
            worldNormalsA1.resizeNoInitialize(nPairs);

            
            btAlignedObjectArray<float4> worldVertsB1;
            worldVertsB1.resizeNoInitialize(vertexFaceCapacity*nPairs);
            btAlignedObjectArray<float4> worldVertsB2;
            worldVertsB2.resizeNoInitialize(vertexFaceCapacity*nPairs);
            
            btAlignedObjectArray<int4> clippingFacesOut;
            clippingFacesOut.resizeNoInitialize(nPairs);
            
            

#ifdef BREAKUP_KERNEL
            bool useGpu = true;
            if (useGpu)
            {
                ///find clipping faces
                {
                    
                    btOpenCLArray<int4> clippingFacesOutGPU(m_context,m_queue);
                    clippingFacesOutGPU.resize(nPairs);
                  
                    btOpenCLArray<float4> worldNormalsAGPU(m_context,m_queue);
                    worldNormalsAGPU.resize(nPairs);
                    
                    btOpenCLArray<float4> worldVertsA1GPU(m_context,m_queue);
                    worldVertsA1GPU.resize(vertexFaceCapacity*nPairs);
                    

                    
                    btOpenCLArray<float4> worldVertsB1GPU(m_context,m_queue);
                    worldVertsB1GPU.resize(vertexFaceCapacity*nPairs);
          
                    
                    btBufferInfoCL bInfo[] = {
                        btBufferInfoCL( pairs->getBufferCL(), true ),
                        btBufferInfoCL( bodyBuf->getBufferCL(),true),
                        btBufferInfoCL( gpuCollidables.getBufferCL(),true),
                        btBufferInfoCL( convexData.getBufferCL(),true),
                        btBufferInfoCL( gpuVertices.getBufferCL(),true),
                        btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                        btBufferInfoCL( gpuFaces.getBufferCL(),true),
                        btBufferInfoCL( gpuIndices.getBufferCL(),true),
                        btBufferInfoCL( sepNormals.getBufferCL()),
                        btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
                        btBufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                        btBufferInfoCL( worldVertsA1GPU.getBufferCL()),
                        btBufferInfoCL( worldNormalsAGPU.getBufferCL()),
                        btBufferInfoCL( worldVertsB1GPU.getBufferCL())
                    };
                    
                    btLauncherCL launcher(m_queue, m_findClippingFacesKernel);
                    launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
                    launcher.setConst( vertexFaceCapacity);
                    launcher.setConst( nPairs  );
                    int num = nPairs;
                    launcher.launch1D( num);
                    clFinish(m_queue);
                    
                    clippingFacesOutGPU.copyToHost(clippingFacesOut);
                    worldVertsB1GPU.copyToHost(worldVertsB1);
                    worldVertsA1GPU.copyToHost(worldVertsA1);
                    worldNormalsAGPU.copyToHost(worldNormalsA1);
                    
                    for (int i=0;i<nPairs;i++)
                    {
                     //   printf("num Faces A=%d\n", clippingFacesOut[i].z);
                       // printf("num Faces A=%d\n", clippingFacesOut[i].z);

                    }
                     
                }

            } else
            {
                BT_PROFILE("findClippingFacesKernel");
                for (int i=0;i<nPairs;i++)
                {
                    g_globalId = i;
                    
                    
                    findClippingFacesKernel((btAlignedObjectArray<int2>&)hostPairs,
                                       (btAlignedObjectArray<struct BodyData>&)hostBodies,
                                       (btAlignedObjectArray<struct btCollidableGpu>&)hostCollidables,
                                       (btAlignedObjectArray<struct ConvexPolyhedronCL>&)hostConvexShapeData,
                                       (btAlignedObjectArray<float4>& )verticesA,
                                       (btAlignedObjectArray<float4>& )uniqueEdges,
                                       (btAlignedObjectArray<btGpuFace>&)faces,
                                       (btAlignedObjectArray<int>&)indices,
                                       (btAlignedObjectArray<float4>&)hostSepNormals,
                                       (btAlignedObjectArray<int>&)hostHasSepAxis,
                                       (btAlignedObjectArray<int4>&)clippingFacesOut,
                                            (btAlignedObjectArray<float4>&) worldVertsA1,
                                            (btAlignedObjectArray<float4>&) worldNormalsA1,
                                            (btAlignedObjectArray<float4>&) worldVertsB1,
                                        vertexFaceCapacity,
                                       nPairs);
                }
            }
            
            {
                BT_PROFILE("clipFacesAndContactReductionKernel");
            for (int i=0;i<nPairs;i++)
            {
                g_globalId = i;
                
                
                
                clipFacesAndContactReductionKernel((btAlignedObjectArray<int2>&)hostPairs,
                                   (btAlignedObjectArray<struct BodyData>&)hostBodies,
                                   (btAlignedObjectArray<float4>&)hostSepNormals,
                                   (btAlignedObjectArray<int>&)hostHasSepAxis,
                                   (btAlignedObjectArray<Contact4>&)hostContactOut,
                                    (btAlignedObjectArray<int4>&) clippingFacesOut,
                                                   (btAlignedObjectArray<float4>&) worldVertsA1,
                                                   (btAlignedObjectArray<float4>&) worldNormalsA1,
                                                   (btAlignedObjectArray<float4>&) worldVertsB1,
                                    (btAlignedObjectArray<float4>&) worldVertsB2,
                                   &nGlobalContactsOut,
                                    vertexFaceCapacity,
                                   nPairs);
            }
            }
            printf("nGlobalContactsOut=%d\n",nGlobalContactsOut);
            
#else//BREAKUP_KERNEL
            for (int i=0;i<nPairs;i++)
            {
                g_globalId = i;
                
                
                clipHullHullKernel((btAlignedObjectArray<int2>&)hostPairs,
                               (btAlignedObjectArray<struct BodyData>&)hostBodies,
                               (btAlignedObjectArray<struct btCollidableGpu>&)hostCollidables,
                               (btAlignedObjectArray<struct ConvexPolyhedronCL>&)hostConvexShapeData,
                               (btAlignedObjectArray<float4>& )verticesA,
                               (btAlignedObjectArray<float4>& )uniqueEdges,
                               (btAlignedObjectArray<btGpuFace>&)faces,
                               (btAlignedObjectArray<int>&)indices,
                               (btAlignedObjectArray<float4>&)hostSepNormals,
                               (btAlignedObjectArray<int>&)hostHasSepAxis,
                               (btAlignedObjectArray<Contact4>&)hostContactOut,
                               &nGlobalContactsOut,
                               nPairs);
                printf("nGlobalContactsOut=%d\n",nGlobalContactsOut);                
            }
#endif//BREAKUP_KERNEL
            
            
            hostContactOut.resize(nGlobalContactsOut);
            if (nGlobalContactsOut != prevGlobalContactOut)
            {
                contactOut->copyFromHost(hostContactOut);
            }
            
            nContacts = nGlobalContactsOut;
            m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
            
            
#else//DEBUG_CPU_CLIP

            
#ifdef BREAKUP_KERNEL

			

			static btAlignedObjectArray<int4> clippingFacesCPU;
            clippingFacesCPU.resize(nPairs);
            
            int vertexFaceCapacity = 64;
            
            static btOpenCLArray<float4> worldVertsB1GPU(m_context,m_queue);
            worldVertsB1GPU.resize(vertexFaceCapacity*nPairs);
          
            static btOpenCLArray<float4> worldVertsB2GPU(m_context,m_queue);
            worldVertsB2GPU.resize(vertexFaceCapacity*nPairs);
            
            
            static btOpenCLArray<int4> clippingFacesOutGPU(m_context,m_queue);
            clippingFacesOutGPU.resize(nPairs);
            
            static btOpenCLArray<float4> worldNormalsAGPU(m_context,m_queue);
            worldNormalsAGPU.resize(nPairs);
            
            static btOpenCLArray<float4> worldVertsA1GPU(m_context,m_queue);
            worldVertsA1GPU.resize(vertexFaceCapacity*nPairs);
            
            
            {
				BT_PROFILE("findClippingFacesKernel");
            btBufferInfoCL bInfo[] = {
                btBufferInfoCL( pairs->getBufferCL(), true ),
                btBufferInfoCL( bodyBuf->getBufferCL(),true),
                btBufferInfoCL( gpuCollidables.getBufferCL(),true),
                btBufferInfoCL( convexData.getBufferCL(),true),
                btBufferInfoCL( gpuVertices.getBufferCL(),true),
                btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                btBufferInfoCL( gpuFaces.getBufferCL(),true), 
                btBufferInfoCL( gpuIndices.getBufferCL(),true),
                btBufferInfoCL( sepNormals.getBufferCL()),
                btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
                btBufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                btBufferInfoCL( worldVertsA1GPU.getBufferCL()),
                btBufferInfoCL( worldNormalsAGPU.getBufferCL()),
                btBufferInfoCL( worldVertsB1GPU.getBufferCL())
            };
            
            btLauncherCL launcher(m_queue, m_findClippingFacesKernel);
            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
            launcher.setConst( vertexFaceCapacity);
            launcher.setConst( nPairs  );
            int num = nPairs;
            launcher.launch1D( num);
            clFinish(m_queue);

            }
            
  
			//clippingFacesOutGPU.copyToHost(clippingFacesCPU);

            ///clip face B against face A, reduce contacts and append them to a global contact array
            if (1)
            {
				BT_PROFILE("clipFacesAndContactReductionKernel");
				//nContacts = m_totalContactsOut.at(0);
				//int h = hasSeparatingNormals.at(0);
				//int4 p = clippingFacesOutGPU.at(0);
                btBufferInfoCL bInfo[] = {
                    btBufferInfoCL( pairs->getBufferCL(), true ),
                    btBufferInfoCL( bodyBuf->getBufferCL(),true),
                    btBufferInfoCL( sepNormals.getBufferCL()),
                    btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
					btBufferInfoCL( contactOut->getBufferCL()),
                    btBufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                    btBufferInfoCL( worldVertsA1GPU.getBufferCL()),
                    btBufferInfoCL( worldNormalsAGPU.getBufferCL()),
                    btBufferInfoCL( worldVertsB1GPU.getBufferCL()),
                    btBufferInfoCL( worldVertsB2GPU.getBufferCL()),
					btBufferInfoCL( m_totalContactsOut.getBufferCL())	
                };
                
                btLauncherCL launcher(m_queue, m_clipFacesAndContactReductionKernel);
                launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
                launcher.setConst(vertexFaceCapacity);
                launcher.setConst( nPairs  );
                int num = nPairs;
                launcher.launch1D( num);
                clFinish(m_queue);
                
				//p = clippingFacesOutGPU.at(0);

                nContacts = m_totalContactsOut.at(0);
            }
            
       
            
#else
            
			btBufferInfoCL bInfo[] = {
				btBufferInfoCL( pairs->getBufferCL(), true ), 
				btBufferInfoCL( bodyBuf->getBufferCL(),true), 
				btBufferInfoCL( gpuCollidables.getBufferCL(),true), 
				btBufferInfoCL( convexData.getBufferCL(),true),
				btBufferInfoCL( gpuVertices.getBufferCL(),true),
				btBufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				btBufferInfoCL( gpuFaces.getBufferCL(),true),
				btBufferInfoCL( gpuIndices.getBufferCL(),true),
				btBufferInfoCL( sepNormals.getBufferCL()),
				btBufferInfoCL( hasSeparatingNormals.getBufferCL()),
				btBufferInfoCL( contactOut->getBufferCL()),
				btBufferInfoCL( m_totalContactsOut.getBufferCL())	
			};
			btLauncherCL launcher(m_queue, m_clipHullHullKernel);
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( nPairs  );
			int num = nPairs;
			launcher.launch1D( num);
			clFinish(m_queue);
		
			nContacts = m_totalContactsOut.at(0);
#endif
            
#endif//DEBUG_CPU_CLIP
            
		}

	} else
	{	
			
		triangleConvexPairsOut.resize(numConcave);
		btAlignedObjectArray<int4> hostTriangleConvexPairsOut;
		triangleConvexPairsOut.copyToHost(hostTriangleConvexPairsOut);
		btAlignedObjectArray<float4> concaveHostNormals;
		concaveSepNormals.resize(numConcave);

		concaveSepNormals.copyToHost(concaveHostNormals);
			

		/*for (int i=0;i<numConcave;i++)
		{
			printf("overlap for pair %d,%d\n",bla[i].x,bla[i].y);
			printf("axis = %f,%f,%f\n",concaveHostNormals[i].x,concaveHostNormals[i].y,concaveHostNormals[i].z);
		}
		*/
//			printf("END\n");

	
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

		
		btAlignedObjectArray<btCollidable> cpuCollidables;
		gpuCollidables.copyToHost(cpuCollidables);


#ifdef __APPLE__
		bool reductionOnGpu = true;
#else
		bool reductionOnGpu = true;
#endif

		//m_hostContactOut.reserve(nPairs);
		m_hostContactOut.resize(nPairs+nContacts);//m_hostContactOut.size()+1);
		int actualContacts = 0;

		
		for (int i=0;i<hostTriangleConvexPairsOut.size();i++)
		{
			int4 pair = hostTriangleConvexPairsOut[i];
			int objectIndexA = pair.x;
			int objectIndexB = pair.y;
			int faceIndex = pair.z;
			float4 sepNormalWorldSpace = concaveHostNormals[i];
			int collidableIndexA = hostBodyBuf[objectIndexA].m_collidableIdx;
			int collidableIndexB = hostBodyBuf[objectIndexB].m_collidableIdx;

			int shapeIndexA = cpuCollidables[collidableIndexA].m_shapeIndex;
			int shapeIndexB = cpuCollidables[collidableIndexB].m_shapeIndex;


			BT_PROFILE("each face");	
			const btGpuFace& face = faces[hostConvexData[shapeIndexA].m_faceOffset+faceIndex];


			//for now we ignore quads, only tris are allowed
			btAssert(face.m_numIndices==3);
                                    
			if (face.m_numIndices==3)
			{
				btAlignedObjectArray<btCollidable> hostCollidablesA;
				btAlignedObjectArray<ConvexPolyhedronCL> hostConvexDataA;
				int collidableIndexA = 0;
				btAlignedObjectArray<btVector3> uniqueEdgesA;
				btAlignedObjectArray<btGpuFace> facesA;
				btAlignedObjectArray<int> indicesA;
                                        
				btAlignedObjectArray<btVector3> verticesA;
                                        
                                        
                                        
                                        
				btCollidable colA;
				colA.m_shapeIndex = 0;
				colA.m_shapeType = CollisionShape::SHAPE_CONVEX_HULL;
				hostCollidablesA.push_back(colA);
                                        
				ConvexPolyhedronCL convexPolyhedronA;

				//add 3 vertices of the triangle
				convexPolyhedronA.m_numVertices = 3;
				convexPolyhedronA.m_vertexOffset = 0;
				btVector3 localCenter(0,0,0);
				btVector3 triMinAabb, triMaxAabb;
				triMinAabb.setValue(1e30,1e30,1e30);
				triMaxAabb.setValue(-1e30,-1e30,-1e30);
										
				{
					BT_PROFILE("extract triangle verts");
					for (int i=0;i<3;i++)
					{
						int index = indices[face.m_indexOffset+i];
						btVector3 vert = vertices[hostConvexData[shapeIndexA].m_vertexOffset+index];
						verticesA.push_back(vert);
						triMinAabb.setMin(vert);
						triMaxAabb.setMax(vert);
						localCenter+=vert;
					}
				}
                                        
				if (1)
				{
					BT_PROFILE("concave-convex actual test");
					int localCC=0;

					//a triangle has 3 unique edges
					convexPolyhedronA.m_numUniqueEdges = 3;
					convexPolyhedronA.m_uniqueEdgesOffset = 0;
					uniqueEdgesA.push_back(verticesA[1]-verticesA[0]);
					uniqueEdgesA.push_back(verticesA[2]-verticesA[1]);
					uniqueEdgesA.push_back(verticesA[0]-verticesA[2]);

					convexPolyhedronA.m_faceOffset = 0;
                                        
					btVector3 normal(face.m_plane.x,face.m_plane.y,face.m_plane.z);
                                        
					//front size of triangle
					{
						btGpuFace gpuFace;
						gpuFace.m_indexOffset=indicesA.size();
						indicesA.push_back(0);
						indicesA.push_back(1);
						indicesA.push_back(2);
						btScalar c = face.m_plane.w;
						gpuFace.m_plane.x = normal[0];
						gpuFace.m_plane.y = normal[1];
						gpuFace.m_plane.z = normal[2];
						gpuFace.m_plane.w = c;
						gpuFace.m_numIndices=3;
						facesA.push_back(gpuFace);
					}

					//back size of triangle
		#if 1
					{
						btGpuFace gpuFace;
						gpuFace.m_indexOffset=indicesA.size();
						indicesA.push_back(2);
						indicesA.push_back(1);
						indicesA.push_back(0);
						btScalar c = (normal.dot(verticesA[0]));
						btScalar c1 = -face.m_plane.w;
						btAssert(c==c1);
						gpuFace.m_plane.x = -normal[0];
						gpuFace.m_plane.y = -normal[1];
						gpuFace.m_plane.z = -normal[2];
						gpuFace.m_plane.w = c;
						gpuFace.m_numIndices=3;
						facesA.push_back(gpuFace);
					}

					bool addEdgePlanes = true;
					if (addEdgePlanes)
					{
						int numVertices=3;
						int prevVertex = numVertices-1;
						for (int i=0;i<numVertices;i++)
						{
							btGpuFace gpuFace;
                                                
							btVector3 v0 = verticesA[i];
							btVector3 v1 = verticesA[prevVertex];
                                                
							btVector3 edgeNormal = (normal.cross(v1-v0)).normalize();
							btScalar c = -edgeNormal.dot(v0);

							gpuFace.m_numIndices = 2;
							gpuFace.m_indexOffset=indicesA.size();
							indicesA.push_back(i);
							indicesA.push_back(prevVertex);
                                                
							gpuFace.m_plane.x = edgeNormal[0];
							gpuFace.m_plane.y = edgeNormal[1];
							gpuFace.m_plane.z = edgeNormal[2];
							gpuFace.m_plane.w = c;
							facesA.push_back(gpuFace);
							prevVertex = i;
						}
					}
		#endif

					convexPolyhedronA.m_numFaces = facesA.size();
					convexPolyhedronA.m_localCenter = localCenter*(1./3.);


                                        
					hostConvexDataA.push_back(convexPolyhedronA);
					int numContactsOut = 0;
					clipHullHullSingle(
						objectIndexA, objectIndexB,
						collidableIndexA, collidableIndexB,
						&hostBodyBuf, 
						&hostShapeBuf,
						contactOut, 
						nContacts, cfg , 
			
						hostConvexDataA,
						hostConvexData,
	
						verticesA, 
						uniqueEdgesA, 
						facesA,
						indicesA,
	
						vertices,
						uniqueEdges,
						faces,
						indices,

						hostCollidablesA,
						cpuCollidables,
						(btVector3&)sepNormalWorldSpace,numContactsOut);
				//	printf("numContactsOut=%d\n",numContactsOut);
				}
			}
		}

		for (int i=0;i<nPairs;i++)
		{
			int indexA = m_hostPairs[i].x;
			int indexB = m_hostPairs[i].y;
			

		
			

			bool validateFindSeparatingAxis = false;//true;
			int collidableA = hostBodyBuf[indexA].m_collidableIdx;
			int collidableB = hostBodyBuf[indexB].m_collidableIdx;

			if (validateFindSeparatingAxis)
			{
			
				btVector3 sepNormalWorldSpace;
				bool foundSepAxis =false;

			
				
				int shapeA = cpuCollidables[collidableA].m_shapeIndex;
				int shapeB = cpuCollidables[collidableB].m_shapeIndex;
				
				BT_PROFILE("findSeparatingAxis");
				foundSepAxis = findSeparatingAxis(
							hostConvexData.at(shapeA), 
							hostConvexData.at(shapeB),
							hostBodyBuf[indexA].m_pos,
							hostBodyBuf[indexA].m_quat,
							hostBodyBuf[indexB].m_pos,
							hostBodyBuf[indexB].m_quat,

							vertices,uniqueEdges,faces,indices,
							vertices,uniqueEdges,faces,indices,
							
							sepNormalWorldSpace);
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
			
			
			if ((cpuCollidables[collidableA].m_shapeType==CollisionShape::SHAPE_CONVEX_HULL)&&(cpuCollidables[collidableB].m_shapeType==CollisionShape::SHAPE_CONVEX_HULL))
			{
				//convex-convex case
				if (hostHasSep[i])
				{
					BT_PROFILE("clipHullAgainstHull");
		
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
                
				
				
					int collidableA = hostBodyBuf[indexA].m_collidableIdx;
					int collidableB = hostBodyBuf[indexB].m_collidableIdx;

				
					int shapeA = cpuCollidables[collidableA].m_shapeIndex;
					int shapeB = cpuCollidables[collidableB].m_shapeIndex;
				

					numContactsOut = clipHullAgainstHull(hostNormals[i], 
						hostConvexData.at(shapeA), 
						hostConvexData.at(shapeB),
									(float4&)trA.getOrigin(), (Quaternion&)trAorn,
									(float4&)trB.getOrigin(), (Quaternion&)trBorn,
									worldVertsB1,worldVertsB2,capacityWorldVerts,
									minDist, maxDist,
									(float4*)&vertices[0],&faces[0],&indices[0],
									(float4*)&vertices[0],&faces[0],&indices[0],
								
									contactsOut,contactCapacity);
				
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
                        m_hostContactOut.resize(nContacts+1);
						Contact4& contact = m_hostContactOut[nContacts];
						contact.m_batchIdx = 0;//i;
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
		}

		
		m_hostContactOut.resize(nContacts);

		if (!reductionOnGpu)
		{
			BT_PROFILE("copyFromHost(m_hostContactOut");
			contactOut->copyFromHost(m_hostContactOut);
		}
	}
}
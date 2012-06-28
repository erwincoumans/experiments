static const char* satKernelsCL= \
"\n"
"\n"
"\n"
"typedef unsigned int u32;\n"
"\n"
"typedef struct\n"
"{\n"
"	float4 m_pos;\n"
"	float4 m_quat;\n"
"	float4 m_linVel;\n"
"	float4 m_angVel;\n"
"\n"
"	u32 m_shapeIdx;\n"
"	u32 m_shapeType;\n"
"	\n"
"	float m_invMass;\n"
"	float m_restituitionCoeff;\n"
"	float m_frictionCoeff;\n"
"} BodyData;\n"
"\n"
"\n"
"typedef struct  \n"
"{\n"
"	float4		m_localCenter;\n"
"	float4		m_extents;\n"
"	float4		mC;\n"
"	float4		mE;\n"
"	float			m_radius;\n"
"	\n"
"	int	m_faceOffset;\n"
"	int m_numFaces;\n"
"\n"
"	int	m_numVertices;\n"
"	int m_vertexOffset;\n"
"\n"
"	int	m_uniqueEdgesOffset;\n"
"	int	m_numUniqueEdges;\n"
"\n"
"} ConvexPolyhedronCL;\n"
"\n"
"typedef struct\n"
"{\n"
"	float4 m_plane;\n"
"	int m_indexOffset;\n"
"	int m_numIndices;\n"
"} btGpuFace;\n"
"\n"
"#define make_float4 (float4)\n"
"\n"
"__inline\n"
"float4 cross3(float4 a, float4 b)\n"
"{\n"
"	return cross(a,b);\n"
"}\n"
"\n"
"__inline\n"
"float dot3F4(float4 a, float4 b)\n"
"{\n"
"	float4 a1 = make_float4(a.xyz,0.f);\n"
"	float4 b1 = make_float4(b.xyz,0.f);\n"
"	return dot(a1, b1);\n"
"}\n"
"\n"
"__inline\n"
"float4 fastNormalize4(float4 v)\n"
"{\n"
"	return fast_normalize(v);\n"
"}\n"
"\n"
"\n"
"///////////////////////////////////////\n"
"//	Quaternion\n"
"///////////////////////////////////////\n"
"\n"
"typedef float4 Quaternion;\n"
"\n"
"__inline\n"
"Quaternion qtMul(Quaternion a, Quaternion b);\n"
"\n"
"__inline\n"
"Quaternion qtNormalize(Quaternion in);\n"
"\n"
"__inline\n"
"float4 qtRotate(Quaternion q, float4 vec);\n"
"\n"
"__inline\n"
"Quaternion qtInvert(Quaternion q);\n"
"\n"
"\n"
"\n"
"\n"
"__inline\n"
"Quaternion qtMul(Quaternion a, Quaternion b)\n"
"{\n"
"	Quaternion ans;\n"
"	ans = cross3( a, b );\n"
"	ans += a.w*b+b.w*a;\n"
"//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);\n"
"	ans.w = a.w*b.w - dot3F4(a, b);\n"
"	return ans;\n"
"}\n"
"\n"
"__inline\n"
"Quaternion qtNormalize(Quaternion in)\n"
"{\n"
"	return fastNormalize4(in);\n"
"//	in /= length( in );\n"
"//	return in;\n"
"}\n"
"__inline\n"
"float4 qtRotate(Quaternion q, float4 vec)\n"
"{\n"
"	Quaternion qInv = qtInvert( q );\n"
"	float4 vcpy = vec;\n"
"	vcpy.w = 0.f;\n"
"	float4 out = qtMul(qtMul(q,vcpy),qInv);\n"
"	return out;\n"
"}\n"
"\n"
"__inline\n"
"Quaternion qtInvert(Quaternion q)\n"
"{\n"
"	return (Quaternion)(-q.xyz, q.w);\n"
"}\n"
"\n"
"__inline\n"
"float4 qtInvRotate(const Quaternion q, float4 vec)\n"
"{\n"
"	return qtRotate( qtInvert( q ), vec );\n"
"}\n"
"\n"
"__inline\n"
"float4 transform(const float4* p, const float4* translation, const Quaternion* orientation)\n"
"{\n"
"	return qtRotate( *orientation, *p ) + (*translation);\n"
"}\n"
"\n"
"\n"
"\n"
"__inline\n"
"float4 normalize3(const float4 a)\n"
"{\n"
"	float4 n = make_float4(a.x, a.y, a.z, 0.f);\n"
"	return fastNormalize4( n );\n"
"}\n"
"\n"
"inline void project(__global const ConvexPolyhedronCL* hull,  const float4 pos, const float4 orn, \n"
"const float4* dir, __global const float4* vertices, float* min, float* max)\n"
"{\n"
"	min[0] = FLT_MAX;\n"
"	max[0] = -FLT_MAX;\n"
"	int numVerts = hull->m_numVertices;\n"
"\n"
"	const float4 localDir = qtInvRotate(orn,*dir);\n"
"	float offset = dot(pos,*dir);\n"
"	for(int i=0;i<numVerts;i++)\n"
"	{\n"
"		float dp = dot(vertices[hull->m_vertexOffset+i],localDir);\n"
"		if(dp < min[0])	\n"
"			min[0] = dp;\n"
"		if(dp > max[0])	\n"
"			max[0] = dp;\n"
"	}\n"
"	if(min[0]>max[0])\n"
"	{\n"
"		float tmp = min[0];\n"
"		min[0] = max[0];\n"
"		max[0] = tmp;\n"
"	}\n"
"	min[0] += offset;\n"
"	max[0] += offset;\n"
"}\n"
"\n"
"\n"
"inline bool TestSepAxis(__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, \n"
"	const float4 posA,const float4 ornA,\n"
"	const float4 posB,const float4 ornB,\n"
"	float4* sep_axis, __global const float4* vertices,float* depth)\n"
"{\n"
"	float Min0,Max0;\n"
"	float Min1,Max1;\n"
"	project(hullA,posA,ornA,sep_axis,vertices, &Min0, &Max0);\n"
"	project(hullB,posB,ornB, sep_axis,vertices, &Min1, &Max1);\n"
"\n"
"	if(Max0<Min1 || Max1<Min0)\n"
"		return false;\n"
"\n"
"	float d0 = Max0 - Min1;\n"
"	float d1 = Max1 - Min0;\n"
"	*depth = d0<d1 ? d0:d1;\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"\n"
"inline bool IsAlmostZero(const float4 v)\n"
"{\n"
"	if(fabs(v.x)>1e-6f || fabs(v.y)>1e-6f || fabs(v.z)>1e-6f)	\n"
"		return false;\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"bool findSeparatingAxisA(	__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, \n"
"	const float4 posA1,\n"
"	const float4 ornA,\n"
"	const float4 posB1,\n"
"	const float4 ornB,\n"
"	__global const float4* vertices, \n"
"	__global const float4* uniqueEdges, \n"
"	__global const btGpuFace* faces,\n"
"	__global const int*  indices,\n"
"	__global volatile float4* sep,\n"
"	float* dmin)\n"
"{\n"
"	int i = get_global_id(0);\n"
"\n"
"	float4 posA = posA1;\n"
"	posA.w = 0.f;\n"
"	float4 posB = posB1;\n"
"	posB.w = 0.f;\n"
"	float4 c0local = hullA->m_localCenter;\n"
"	float4 c0 = transform(&c0local, &posA, &ornA);\n"
"	float4 c1local = hullB->m_localCenter;\n"
"	float4 c1 = transform(&c1local,&posB,&ornB);\n"
"	const float4 DeltaC2 = c0 - c1;\n"
"\n"
"\n"
"	int curPlaneTests=0;\n"
"\n"
"	{\n"
"		int numFacesA = hullA->m_numFaces;\n"
"		// Test normals from hullA\n"
"		for(int i=0;i<numFacesA;i++)\n"
"		{\n"
"			const float4 normal = faces[hullA->m_faceOffset+i].m_plane;\n"
"			float4 faceANormalWS = qtRotate(ornA,normal);\n"
"	\n"
"			if (dot3F4(DeltaC2,faceANormalWS)<0)\n"
"				continue;\n"
"	\n"
"			curPlaneTests++;\n"
"	\n"
"			float d;\n"
"			if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,&faceANormalWS, vertices,&d))\n"
"				return false;\n"
"	\n"
"			if(d<*dmin)\n"
"			{\n"
"				*dmin = d;\n"
"				*sep = faceANormalWS;\n"
"			}\n"
"		}\n"
"	}\n"
"\n"
"	const float4 deltaC = posB - posA;\n"
"	if((dot3F4(deltaC,*sep))>0.0f)\n"
"	{\n"
"		*sep = -(*sep);\n"
"	}\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"bool findSeparatingAxisB(	__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, \n"
"	const float4 posA1,\n"
"	const float4 ornA,\n"
"	const float4 posB1,\n"
"	const float4 ornB,\n"
"	__global const float4* vertices, \n"
"	__global const float4* uniqueEdges, \n"
"	__global const btGpuFace* faces,\n"
"	__global const int*  indices,\n"
"	__global volatile float4* sep,\n"
"	float* dmin)\n"
"{\n"
"	int i = get_global_id(0);\n"
"\n"
"	float4 posA = posA1;\n"
"	posA.w = 0.f;\n"
"	float4 posB = posB1;\n"
"	posB.w = 0.f;\n"
"	float4 c0local = hullA->m_localCenter;\n"
"	float4 c0 = transform(&c0local, &posA, &ornA);\n"
"	float4 c1local = hullB->m_localCenter;\n"
"	float4 c1 = transform(&c1local,&posB,&ornB);\n"
"	const float4 DeltaC2 = c0 - c1;\n"
"\n"
"\n"
"	int curPlaneTests=0;\n"
"\n"
"	\n"
"\n"
"	int numFacesB = hullB->m_numFaces;\n"
"	// Test normals from hullB\n"
"	for(int i=0;i<numFacesB;i++)\n"
"	{\n"
"		float4 normal = faces[hullB->m_faceOffset+i].m_plane;\n"
"		const float4 WorldNormal = qtRotate(ornB, normal);\n"
"\n"
"		if (dot3F4(DeltaC2,WorldNormal)<0)\n"
"			continue;\n"
"\n"
"		curPlaneTests++;\n"
"\n"
"		float d;\n"
"		if(!TestSepAxis(hullA, hullB,posA,ornA,posB,ornB,&WorldNormal,vertices,&d))\n"
"			return false;\n"
"\n"
"		if(d<*dmin)\n"
"		{\n"
"			*dmin = d;\n"
"			*sep = WorldNormal;\n"
"		}\n"
"	}\n"
"\n"
"	const float4 deltaC = posB - posA;\n"
"	if((dot3F4(deltaC,*sep))>0.0f)\n"
"	{\n"
"		*sep = -(*sep);\n"
"	}\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"bool findSeparatingAxisEdgeEdge(	__global const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, \n"
"	const float4 posA1,\n"
"	const float4 ornA,\n"
"	const float4 posB1,\n"
"	const float4 ornB,\n"
"	__global const float4* vertices, \n"
"	__global const float4* uniqueEdges, \n"
"	__global const btGpuFace* faces,\n"
"	__global const int*  indices,\n"
"	__global volatile float4* sep,\n"
"	float* dmin)\n"
"{\n"
"	int i = get_global_id(0);\n"
"\n"
"	float4 posA = posA1;\n"
"	posA.w = 0.f;\n"
"	float4 posB = posB1;\n"
"	posB.w = 0.f;\n"
"	float4 c0local = hullA->m_localCenter;\n"
"	float4 c0 = transform(&c0local, &posA, &ornA);\n"
"	float4 c1local = hullB->m_localCenter;\n"
"	float4 c1 = transform(&c1local,&posB,&ornB);\n"
"	const float4 DeltaC2 = c0 - c1;\n"
"\n"
"\n"
"	int curPlaneTests=0;\n"
"\n"
"	int curEdgeEdge = 0;\n"
"	// Test edges\n"
"	for(int e0=0;e0<hullA->m_numUniqueEdges;e0++)\n"
"	{\n"
"		const float4 edge0 = uniqueEdges[hullA->m_uniqueEdgesOffset+e0];\n"
"		float4 edge0World = qtRotate(ornA,edge0);\n"
"\n"
"		for(int e1=0;e1<hullB->m_numUniqueEdges;e1++)\n"
"		{\n"
"			const float4 edge1 = uniqueEdges[hullB->m_uniqueEdgesOffset+e1];\n"
"			float4 edge1World = qtRotate(ornB,edge1);\n"
"\n"
"\n"
"			float4 crossje = cross3(edge0World,edge1World);\n"
"\n"
"			curEdgeEdge++;\n"
"			if(!IsAlmostZero(crossje))\n"
"			{\n"
"				crossje = normalize3(crossje);\n"
"				if (dot3F4(DeltaC2,crossje)<0)\n"
"					continue;\n"
"\n"
"				float dist;\n"
"				bool result = true;\n"
"				{\n"
"					float Min0,Max0;\n"
"					float Min1,Max1;\n"
"					project(hullA,posA,ornA,&crossje,vertices, &Min0, &Max0);\n"
"					project(hullB,posB,ornB,&crossje,vertices, &Min1, &Max1);\n"
"				\n"
"					if(Max0<Min1 || Max1<Min0)\n"
"						result = false;\n"
"				\n"
"					float d0 = Max0 - Min1;\n"
"					float d1 = Max1 - Min0;\n"
"					dist = d0<d1 ? d0:d1;\n"
"					result = true;\n"
"\n"
"				}\n"
"				\n"
"\n"
"				if(dist<*dmin)\n"
"				{\n"
"					*dmin = dist;\n"
"					*sep = crossje;\n"
"				}\n"
"			}\n"
"		}\n"
"\n"
"	}\n"
"\n"
"	const float4 deltaC = posB - posA;\n"
"	if((dot3F4(deltaC,*sep))>0.0f)\n"
"	{\n"
"		*sep = -(*sep);\n"
"	}\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"// work-in-progress\n"
"__kernel void   findSeparatingAxisKernel( __global const int2* pairs, \n"
"																					__global const BodyData* rigidBodies, \n"
"																					__global const ConvexPolyhedronCL* convexShapes, \n"
"																					__global const float4* vertices,\n"
"																					__global const float4* uniqueEdges,\n"
"																					__global const btGpuFace* faces,\n"
"																					__global const int* indices,\n"
"																					__global volatile float4* separatingNormals,\n"
"																					__global volatile int* hasSeparatingAxis,\n"
"																					int numPairs)\n"
"{\n"
"\n"
"	int i = get_global_id(0);\n"
"	if (i<numPairs)\n"
"	{\n"
"		int bodyIndexA = pairs[i].x;\n"
"		int bodyIndexB = pairs[i].y;\n"
"		int shapeIndexA = rigidBodies[bodyIndexA].m_shapeIdx;\n"
"		int shapeIndexB = rigidBodies[bodyIndexB].m_shapeIdx;\n"
"//once the broadphase avoids static-static pairs, we can remove this test\n"
"		if ((rigidBodies[bodyIndexA].m_invMass==0) &&(rigidBodies[bodyIndexB].m_invMass==0))\n"
"			return;\n"
"\n"
"		int numFacesA = convexShapes[shapeIndexA].m_numFaces;\n"
"\n"
"		float dmin = FLT_MAX;\n"
"\n"
"		\n"
"		bool sepA = findSeparatingAxisA(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,\n"
"																								rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,vertices,uniqueEdges,faces,\n"
"																								indices,&separatingNormals[i],&dmin);\n"
"		hasSeparatingAxis[i] = 4;\n"
"		if (!sepA)\n"
"		{\n"
"			hasSeparatingAxis[i] = 0;\n"
"		} else\n"
"		{\n"
"			bool sepB = findSeparatingAxisB(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,\n"
"																									rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,vertices,uniqueEdges,faces,\n"
"																									indices,&separatingNormals[i],&dmin);\n"
"\n"
"			if (!sepB)\n"
"			{\n"
"				hasSeparatingAxis[i] = 0;\n"
"			} else\n"
"			{\n"
"				bool sepEE = findSeparatingAxisEdgeEdge(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,\n"
"																									rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,vertices,uniqueEdges,faces,\n"
"																									indices,&separatingNormals[i],&dmin);\n"
"				if (!sepEE)\n"
"				{\n"
"					hasSeparatingAxis[i] = 0;\n"
"				} else\n"
"				{\n"
"					hasSeparatingAxis[i] = 1;\n"
"				}\n"
"			}\n"
"		}\n"
"		\n"
"	}\n"
"}\n"
;
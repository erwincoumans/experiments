/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#include "OpenGLInclude.h"

#include "CLPhysicsDemo.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "DemoSettings.h"
#include "../basic_initialize/btOpenCLUtils.h"
#include "../opengl_interop/btOpenCLGLInteropBuffer.h"
#include "../broadphase_benchmark/findPairsOpenCL.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btQuaternion.h"
#include "LinearMath/btMatrix3x3.h"
#include "../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h"
#include "../opencl/gpu_rigidbody_pipeline/btConvexUtility.h"
#include "../../dynamics/basic_demo/ConvexHeightFieldShape.h"
//#define USE_GRID_BROADPHASE
#ifdef USE_GRID_BROADPHASE
#include "../broadphase_benchmark/btGridBroadphaseCl.h"
#else
#include "btGpuSapBroadphase.h"
#endif //USE_GRID_BROADPHASE

#include "../broadphase_benchmark/btAabbHost.h"
#include "LinearMath/btQuickprof.h"


#define MSTRINGIFY(A) #A
static char* interopKernelString = 
#include "../broadphase_benchmark/integrateKernel.cl"

#define INTEROPKERNEL_SRC_PATH "../../opencl/broadphase_benchmark/integrateKernel.cl"
	
cl_kernel g_integrateTransformsKernel;



bool runOpenCLKernels = true;


btGpuNarrowphaseAndSolver* narrowphaseAndSolver = 0;
ConvexHeightField* s_convexHeightField = 0 ;
btOpenCLGLInteropBuffer* g_interopBuffer = 0;

extern GLuint               cube_vbo;
extern int VBOsize;

cl_mem clBuffer=0;
char* hostPtr=0;
cl_bool blocking=  CL_TRUE;



btFindPairsIO gFpIO;

cl_context			g_cxMainContext;
cl_command_queue	g_cqCommandQue;
cl_device_id		g_device;





struct InternalData
{
	btOpenCLArray<btVector3>* m_linVelBuf;
	btOpenCLArray<btVector3>* m_angVelBuf;
	btOpenCLArray<float>* m_bodyTimes;
	bool	m_useInterop;
#ifdef USE_GRID_BROADPHASE
	btGridBroadphaseCl* m_Broadphase;
#else
	btGpuSapBroadphase* m_Broadphase;
#endif //USE_GRID_BROADPHASE

	btOpenCLArray<btAABBHost>* m_localShapeAABB;

	btVector3*	m_linVelHost;
	btVector3*	m_angVelHost;
	float*		m_bodyTimesHost;

	InternalData():m_linVelBuf(0),m_angVelBuf(0),m_bodyTimes(0),m_useInterop(0),m_Broadphase(0)
	{
		m_linVelHost= new btVector3[MAX_CONVEX_BODIES_CL];
		m_angVelHost = new btVector3[MAX_CONVEX_BODIES_CL];
		m_bodyTimesHost = new float[MAX_CONVEX_BODIES_CL];
		for (int i=0;i<MAX_CONVEX_BODIES_CL;i++)
		{
			m_linVelHost[i].setZero();
			m_angVelHost[i].setZero();
			m_bodyTimesHost[i] = 0.f;
		}
	}
	~InternalData()
	{
		delete[] m_linVelHost;
		delete[] m_angVelHost;
		delete[] m_bodyTimesHost;

	}
};


void InitCL(int preferredDeviceIndex, int preferredPlatformIndex, bool useInterop)
{
	void* glCtx=0;
	void* glDC = 0;

#ifdef _WIN32
	glCtx = wglGetCurrentContext();
#else //!_WIN32
	GLXContext glCtx = glXGetCurrentContext();
#endif //!_WIN32
	glDC = wglGetCurrentDC();

	int ciErrNum = 0;
#ifdef CL_PLATFORM_INTEL
	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
#else
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
#endif

	

	if (useInterop)
	{
		g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, glCtx, glDC);
	} else
	{
		g_cxMainContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, 0,0,preferredDeviceIndex, preferredPlatformIndex);
	}


	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	int numDev = btOpenCLUtils::getNumDevices(g_cxMainContext);

	if (numDev>0)
	{
		g_device= btOpenCLUtils::getDevice(g_cxMainContext,0);
		btOpenCLUtils::printDeviceInfo(g_device);
		g_cqCommandQue = clCreateCommandQueue(g_cxMainContext, g_device, 0, &ciErrNum);
		oclCHECKERROR(ciErrNum, CL_SUCCESS);
	}

}




CLPhysicsDemo::CLPhysicsDemo(Win32OpenGLWindow*	renderer)
{
	m_numCollisionShapes=0;
	m_numPhysicsInstances=0;

	m_data = new InternalData;
}

CLPhysicsDemo::~CLPhysicsDemo()
{

}


void CLPhysicsDemo::writeBodiesToGpu()
{
	if (narrowphaseAndSolver)
		narrowphaseAndSolver->writeAllBodiesToGpu();
}

int		CLPhysicsDemo::registerCollisionShape(const float* vertices, int strideInBytes, int numVertices, const float* scaling)
{
	btAlignedObjectArray<btVector3> verts;
	
	unsigned char* vts = (unsigned char*) vertices;
	for (int i=0;i<numVertices;i++)
	{
		float* vertex = (float*) &vts[i*strideInBytes];
		verts.push_back(btVector3(vertex[0]*scaling[0],vertex[1]*scaling[1],vertex[2]*scaling[2]));
	}

	btConvexUtility* utilPtr = new btConvexUtility();
	bool merge = true;
	utilPtr->initializePolyhedralFeatures(verts,merge);

	int numFaces= utilPtr->m_faces.size();
	float4* eqn = new float4[numFaces];
	for (int i=0;i<numFaces;i++)
	{
		eqn[i].x = utilPtr->m_faces[i].m_plane[0];
		eqn[i].y = utilPtr->m_faces[i].m_plane[1];
		eqn[i].z = utilPtr->m_faces[i].m_plane[2];
		eqn[i].w = utilPtr->m_faces[i].m_plane[3];
	}
	printf("numFaces = %d\n", numFaces);


	s_convexHeightField = new ConvexHeightField(eqn,numFaces);

	int shapeIndex=-1;

	if (narrowphaseAndSolver)
		shapeIndex = narrowphaseAndSolver->registerShape(s_convexHeightField,utilPtr);

	if (shapeIndex>=0)
	{
		btAABBHost aabbMin, aabbMax;
		aabbMin.fx = s_convexHeightField->m_aabb.m_min.x;
		aabbMin.fy = s_convexHeightField->m_aabb.m_min.y;
		aabbMin.fz= s_convexHeightField->m_aabb.m_min.z;
		aabbMin.uw = shapeIndex;

		aabbMax.fx = s_convexHeightField->m_aabb.m_max.x;
		aabbMax.fy = s_convexHeightField->m_aabb.m_max.y;
		aabbMax.fz= s_convexHeightField->m_aabb.m_max.z;
		aabbMax.uw = shapeIndex;

		m_data->m_localShapeAABB->copyFromHostPointer(&aabbMin,1,shapeIndex*2);
		m_data->m_localShapeAABB->copyFromHostPointer(&aabbMax,1,shapeIndex*2+1);
		clFinish(g_cqCommandQue);
	}

	m_numCollisionShapes++;
	delete[] eqn;
	return shapeIndex;
}

int		CLPhysicsDemo::registerPhysicsInstance(float mass, const float* position, const float* orientation, int collisionShapeIndex, void* userPointer)
{
	btVector3 aabbMin(position[0],position[0],position[0]);
	btVector3 aabbMax = aabbMin;
	aabbMin -= btVector3(1.f,1.f,1.f)*0.1f;
	aabbMax += btVector3(1.f,1.f,1.f)*0.1f;

	if (collisionShapeIndex>=0)
	{
		//btBroadphaseProxy* proxy = m_data->m_Broadphase->createProxy(aabbMin,aabbMax,collisionShapeIndex,userPointer,1,1,0,0);//m_dispatcher);
		m_data->m_Broadphase->createProxy(aabbMin,aabbMax,collisionShapeIndex,userPointer,1,1);//m_dispatcher);
	}
			
	bool writeToGpu = false;
	int bodyIndex = -1;

	if (narrowphaseAndSolver)
		bodyIndex = narrowphaseAndSolver->registerRigidBody(collisionShapeIndex,mass,position,orientation,writeToGpu);

	m_numPhysicsInstances++;
	return bodyIndex;
}



void	CLPhysicsDemo::init(int preferredDevice, int preferredPlatform, bool useInterop)
{
	
	InitCL(-1,-1,useInterop);


	//adl::Solver<adl::TYPE_CL>::allocate(g_deviceCL->allocate(
	m_data->m_linVelBuf = new btOpenCLArray<btVector3>(g_cxMainContext,g_cqCommandQue,MAX_CONVEX_BODIES_CL,false);
	m_data->m_angVelBuf = new btOpenCLArray<btVector3>(g_cxMainContext,g_cqCommandQue,MAX_CONVEX_BODIES_CL,false);
	m_data->m_bodyTimes = new btOpenCLArray<float>(g_cxMainContext,g_cqCommandQue,MAX_CONVEX_BODIES_CL,false);
	
	m_data->m_localShapeAABB = new btOpenCLArray<btAABBHost>(g_cxMainContext,g_cqCommandQue,MAX_CONVEX_SHAPES_CL,false);
	

	writeVelocitiesToGpu();


	narrowphaseAndSolver = new btGpuNarrowphaseAndSolver(g_cxMainContext,g_device,g_cqCommandQue);

	
	
	int maxObjects = btMax(256,MAX_CONVEX_BODIES_CL);
	int maxPairsSmallProxy = 32;
	

#ifdef USE_GRID_BROADPHASE
	btOverlappingPairCache* overlappingPairCache=0;
	m_data->m_Broadphase = new btGridBroadphaseCl(overlappingPairCache,btVector3(4.f, 4.f, 4.f), 128, 128, 128,maxObjects, maxObjects, maxPairsSmallProxy, 100.f, 128,
		g_cxMainContext ,g_device,g_cqCommandQue);
#else //USE_GRID_BROADPHASE
	m_data->m_Broadphase = new btGpuSapBroadphase(g_cxMainContext ,g_device,g_cqCommandQue);//overlappingPairCache,btVector3(4.f, 4.f, 4.f), 128, 128, 128,maxObjects, maxObjects, maxPairsSmallProxy, 100.f, 128,
		//g_cxMainContext ,g_device,g_cqCommandQue);
	
#endif//USE_GRID_BROADPHASE
	

	cl_program prog = btOpenCLUtils::compileCLProgramFromString(g_cxMainContext,g_device,interopKernelString,0,"",INTEROPKERNEL_SRC_PATH);
	g_integrateTransformsKernel = btOpenCLUtils::compileCLKernelFromString(g_cxMainContext, g_device,interopKernelString, "integrateTransformsKernel" ,0,prog);
	

	initFindPairs(gFpIO, g_cxMainContext, g_device, g_cqCommandQue, MAX_CONVEX_BODIES_CL);

	


}
	


void CLPhysicsDemo::writeVelocitiesToGpu()
{
	m_data->m_linVelBuf->copyFromHostPointer(m_data->m_linVelHost,MAX_CONVEX_BODIES_CL);
	m_data->m_angVelBuf->copyFromHostPointer(m_data->m_angVelHost,MAX_CONVEX_BODIES_CL);
	m_data->m_bodyTimes->copyFromHostPointer(m_data->m_bodyTimesHost,MAX_CONVEX_BODIES_CL);
	clFinish(g_cqCommandQue);
}


void CLPhysicsDemo::setupInterop()
{
	m_data->m_useInterop = true;

	g_interopBuffer = new btOpenCLGLInteropBuffer(g_cxMainContext,g_cqCommandQue,cube_vbo);
	clFinish(g_cqCommandQue);
}

void	CLPhysicsDemo::cleanup()
{
	delete narrowphaseAndSolver;

	delete m_data->m_linVelBuf;
	delete m_data->m_angVelBuf;
	delete m_data->m_bodyTimes;
	delete m_data->m_localShapeAABB;

	delete m_data->m_Broadphase;


	m_data=0;
	delete g_interopBuffer;
	delete s_convexHeightField;
}





void	CLPhysicsDemo::stepSimulation()
{
	BT_PROFILE("simulationLoop");
	
	
	cl_int ciErrNum = CL_SUCCESS;


	if(m_data->m_useInterop)
	{
		clBuffer = g_interopBuffer->getCLBUffer();
		BT_PROFILE("clEnqueueAcquireGLObjects");
		ciErrNum = clEnqueueAcquireGLObjects(g_cqCommandQue, 1, &clBuffer, 0, 0, NULL);
		clFinish(g_cqCommandQue);
	} else
	{

		glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);
		glFlush();

		BT_PROFILE("glMapBuffer and clEnqueueWriteBuffer");

		blocking=  CL_TRUE;
		hostPtr=  (char*)glMapBuffer( GL_ARRAY_BUFFER,GL_READ_WRITE);//GL_WRITE_ONLY
		if (!clBuffer)
		{
			clBuffer = clCreateBuffer(g_cxMainContext, CL_MEM_READ_WRITE, VBOsize, 0, &ciErrNum);
		} 
		clFinish(g_cqCommandQue);
			oclCHECKERROR(ciErrNum, CL_SUCCESS);

		ciErrNum = clEnqueueWriteBuffer (	g_cqCommandQue,
 			clBuffer,
 			blocking,
 			0,
 			VBOsize,
 			hostPtr,0,0,0
		);
		clFinish(g_cqCommandQue);
	}



	oclCHECKERROR(ciErrNum, CL_SUCCESS);
	if (runOpenCLKernels && m_numPhysicsInstances)
	{

		gFpIO.m_numObjects = m_numPhysicsInstances;
		gFpIO.m_positionOffset = SHAPE_VERTEX_BUFFER_SIZE/4;
		gFpIO.m_clObjectsBuffer = clBuffer;
		gFpIO.m_dAABB = m_data->m_Broadphase->getAabbBuffer();
		gFpIO.m_dlocalShapeAABB = (cl_mem)m_data->m_localShapeAABB->getBufferCL();
		gFpIO.m_numOverlap = 0;
		{
			BT_PROFILE("setupGpuAabbs");
			setupGpuAabbsFull(gFpIO,narrowphaseAndSolver->getBodiesGpu() );
		}
		if (1)
		{
			BT_PROFILE("calculateOverlappingPairs");
			m_data->m_Broadphase->calculateOverlappingPairs();
			gFpIO.m_dAllOverlappingPairs = m_data->m_Broadphase->getOverlappingPairBuffer();
			gFpIO.m_numOverlap = m_data->m_Broadphase->getNumOverlap();
		}
		
		//printf("gFpIO.m_numOverlap = %d\n",gFpIO.m_numOverlap );
		if (gFpIO.m_numOverlap>=0 && gFpIO.m_numOverlap<MAX_BROADPHASE_COLLISION_CL)
		{
			colorPairsOpenCL(gFpIO);

			if (1)
			{
				{
					//BT_PROFILE("setupBodies");
					if (narrowphaseAndSolver)
						setupBodies(gFpIO, m_data->m_linVelBuf->getBufferCL(), m_data->m_angVelBuf->getBufferCL(), narrowphaseAndSolver->getBodiesGpu(), narrowphaseAndSolver->getBodyInertiasGpu());
				}
				
				{
					BT_PROFILE("computeContactsAndSolver");
					if (narrowphaseAndSolver)
						narrowphaseAndSolver->computeContactsAndSolver(gFpIO.m_dAllOverlappingPairs,gFpIO.m_numOverlap);
				}

				{
					BT_PROFILE("copyBodyVelocities");
					if (narrowphaseAndSolver)
						copyBodyVelocities(gFpIO, m_data->m_linVelBuf->getBufferCL(), m_data->m_angVelBuf->getBufferCL(), narrowphaseAndSolver->getBodiesGpu(), narrowphaseAndSolver->getBodyInertiasGpu());
				}
			}

		} else
		{
			printf("error, gFpIO.m_numOverlap = %d\n",gFpIO.m_numOverlap);
			btAssert(0);
		}


		{
			BT_PROFILE("integrateTransforms");

			if (runOpenCLKernels)
			{
				int numObjects = m_numPhysicsInstances;
				int offset = SHAPE_VERTEX_BUFFER_SIZE/4;

				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 0, sizeof(int), &offset);
				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 1, sizeof(int), &numObjects);
				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 2, sizeof(cl_mem), (void*)&clBuffer );

				cl_mem lv = m_data->m_linVelBuf->getBufferCL();
				cl_mem av = m_data->m_angVelBuf->getBufferCL();
				cl_mem btimes = m_data->m_bodyTimes->getBufferCL();

				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 3, sizeof(cl_mem), (void*)&lv);
				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 4, sizeof(cl_mem), (void*)&av);
				ciErrNum = clSetKernelArg(g_integrateTransformsKernel, 5, sizeof(cl_mem), (void*)&btimes);
					
					
					

				size_t workGroupSize = 64;
				size_t	numWorkItems = workGroupSize*((m_numPhysicsInstances + (workGroupSize)) / workGroupSize);
				
				if (workGroupSize>numWorkItems)
					workGroupSize=numWorkItems;

				ciErrNum = clEnqueueNDRangeKernel(g_cqCommandQue, g_integrateTransformsKernel, 1, NULL, &numWorkItems, &workGroupSize,0 ,0 ,0);
				oclCHECKERROR(ciErrNum, CL_SUCCESS);
			}
		}
			

	}

	if(m_data->m_useInterop)
	{
		BT_PROFILE("clEnqueueReleaseGLObjects");
		ciErrNum = clEnqueueReleaseGLObjects(g_cqCommandQue, 1, &clBuffer, 0, 0, 0);
		clFinish(g_cqCommandQue);
	}
	else
	{
		BT_PROFILE("clEnqueueReadBuffer clReleaseMemObject and glUnmapBuffer");
		ciErrNum = clEnqueueReadBuffer (	g_cqCommandQue,
 		clBuffer,
 		blocking,
 		0,
 		VBOsize,
 		hostPtr,0,0,0);

		//clReleaseMemObject(clBuffer);
		clFinish(g_cqCommandQue);
		glUnmapBuffer( GL_ARRAY_BUFFER);
		glFlush();
	}

	oclCHECKERROR(ciErrNum, CL_SUCCESS);


	if (runOpenCLKernels)
	{
		BT_PROFILE("clFinish");
		clFinish(g_cqCommandQue);
	}

	
}
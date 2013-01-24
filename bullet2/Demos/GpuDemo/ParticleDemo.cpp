#include "ParticleDemo.h"

#include "../../rendering/rendertest/GLInstancingRenderer.h"
#include "../../rendering/rendertest/ShapeData.h"
#include "../../opencl/basic_initialize/btOpenCLUtils.h"

#define MSTRINGIFY(A) #A
static char* particleKernelsString = 
#include "ParticleKernels.cl"

#define INTEROPKERNEL_SRC_PATH "../../bullet2/Demos/GpuDemo/ParticleKernels.cl"

#include "../../rendering/rendertest/OpenGLInclude.h"
#include "../../rendering/rendertest/GLInstanceRendererInternalData.h"
#include "../../opencl/broadphase_benchmark/btLauncherCL.h"

//1000000 particles
//#define NUM_PARTICLES_X 100
//#define NUM_PARTICLES_Y 100
//#define NUM_PARTICLES_Z 100

//512k particles
//#define NUM_PARTICLES_X 80
//#define NUM_PARTICLES_Y 80
//#define NUM_PARTICLES_Z 80

//256k particles
//#define NUM_PARTICLES_X 60
//#define NUM_PARTICLES_Y 60
//#define NUM_PARTICLES_Z 60

//27k particles
#define NUM_PARTICLES_X 30
#define NUM_PARTICLES_Y 30
#define NUM_PARTICLES_Z 30

struct myfloat4
{
	float	m_x;
	float	m_y;
	float	m_z;
	float	m_w;
};

struct ParticleInternalData
{
	cl_context m_clContext;
	cl_device_id m_clDevice;
	cl_command_queue m_clQueue;
	cl_kernel m_updatePositionsKernel;

	cl_mem		m_clPositionBuffer;

	btAlignedObjectArray<myfloat4> m_velocitiesCPU;
	btOpenCLArray<myfloat4>*	m_velocitiesGPU;

	bool m_clInitialized;

	ParticleInternalData()
		:m_clInitialized(false),
		m_clPositionBuffer(0),
		m_velocitiesGPU(0)
	{

	}

	char*	m_clDeviceName;

};


ParticleDemo::ParticleDemo()
:m_instancingRenderer(0)
{
	m_data = new ParticleInternalData;
}

ParticleDemo::~ParticleDemo()
{
	exitCL();

	delete m_data;

}

void ParticleDemo::exitCL()
{
	if (m_data->m_clInitialized)
	{
		m_data->m_clInitialized = false;
		clReleaseCommandQueue(m_data->m_clQueue);
		clReleaseKernel(m_data->m_updatePositionsKernel);
		clReleaseContext(m_data->m_clContext);
	}
}

void ParticleDemo::initCL(int preferredDeviceIndex, int preferredPlatformIndex)
{
	void* glCtx=0;
	void* glDC = 0;


    
	int ciErrNum = 0;
//#ifdef CL_PLATFORM_INTEL
//	cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
//#else
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
//#endif

	

//	if (useInterop)
//	{
//		m_data->m_clContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, glCtx, glDC);
//	} else
	{
		m_data->m_clContext = btOpenCLUtils::createContextFromType(deviceType, &ciErrNum, 0,0,preferredDeviceIndex, preferredPlatformIndex);
	}


	oclCHECKERROR(ciErrNum, CL_SUCCESS);

	int numDev = btOpenCLUtils::getNumDevices(m_data->m_clContext);

	if (numDev>0)
	{
		m_data->m_clDevice= btOpenCLUtils::getDevice(m_data->m_clContext,0);
		m_data->m_clQueue = clCreateCommandQueue(m_data->m_clContext, m_data->m_clDevice, 0, &ciErrNum);
		oclCHECKERROR(ciErrNum, CL_SUCCESS);
        
        btOpenCLUtils::printDeviceInfo(m_data->m_clDevice);
		btOpenCLDeviceInfo info;
		btOpenCLUtils::getDeviceInfo(m_data->m_clDevice,&info);
		m_data->m_clDeviceName = info.m_deviceName;
		m_data->m_clInitialized = true;

	}

}


void ParticleDemo::setupScene(const ConstructionInfo& ci)
{

	initCL(ci.preferredOpenCLDeviceIndex,ci.preferredOpenCLPlatformIndex);
	int numParticles = NUM_PARTICLES_X*NUM_PARTICLES_Y*NUM_PARTICLES_Z;

	m_data->m_velocitiesGPU = new btOpenCLArray<myfloat4>(m_data->m_clContext,m_data->m_clQueue,numParticles);
	m_data->m_velocitiesCPU.resize(numParticles);
	for (int i=0;i<numParticles;i++)
	{
		m_data->m_velocitiesCPU[i].m_x = 0.2f;
		m_data->m_velocitiesCPU[i].m_y = 0.f;
		m_data->m_velocitiesCPU[i].m_z = 0.f;
		m_data->m_velocitiesCPU[i].m_w = 0.f;
	}
	m_data->m_velocitiesGPU->copyFromHost(m_data->m_velocitiesCPU);

	cl_int pErrNum;

	cl_program prog = btOpenCLUtils::compileCLProgramFromString(m_data->m_clContext,m_data->m_clDevice,particleKernelsString,0,"",INTEROPKERNEL_SRC_PATH);
	m_data->m_updatePositionsKernel = btOpenCLUtils::compileCLKernelFromString(m_data->m_clContext, m_data->m_clDevice,particleKernelsString, "updatePositionsKernel" ,&pErrNum,prog);
	oclCHECKERROR(pErrNum, CL_SUCCESS);


	m_instancingRenderer = ci.m_instancingRenderer;

	int strideInBytes = 9*sizeof(float);
	int numVertices = sizeof(point_sphere_vertices)/strideInBytes;
	int numIndices = sizeof(point_sphere_indices)/sizeof(int);
	int shapeId = m_instancingRenderer->registerShape(&point_sphere_vertices[0],numVertices,point_sphere_indices,numIndices,BT_GL_POINTS);

	float position[4] = {0,0,0,0};
	float quaternion[4] = {0,0,0,1};
	float color[4]={1,0,0,1};
	float scaling[4] = {1,1,1,1};

	for (int x=0;x<NUM_PARTICLES_X;x++)
	{
		for (int y=0;y<NUM_PARTICLES_Y;y++)
		{
			for (int z=0;z<NUM_PARTICLES_Z;z++)
			{
				position[0] = x*4;
				position[1] = y*4;
				position[2] = z*4;

				color[0] = float(x)/float(NUM_PARTICLES_X);
				color[1] = float(y)/float(NUM_PARTICLES_Y);
				color[2] = float(z)/float(NUM_PARTICLES_Z);

				int id = m_instancingRenderer->registerGraphicsInstance(shapeId,position,quaternion,color,scaling);
			}
		}
	}

	m_instancingRenderer->writeTransforms();

}

void	ParticleDemo::initPhysics(const ConstructionInfo& ci)
{
	setupScene(ci);
}

void	ParticleDemo::exitPhysics()
{
}

void	ParticleDemo::renderScene()
{
	
	if (m_instancingRenderer)
	{
		m_instancingRenderer->RenderScene();
	}

}


void ParticleDemo::clientMoveAndDisplay()
{
	int numParticles = NUM_PARTICLES_X*NUM_PARTICLES_Y*NUM_PARTICLES_Z;
	GLuint vbo = m_instancingRenderer->getInternalData()->m_vbo;
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glFlush();

	int posArraySize = numParticles*sizeof(float)*4;

	cl_bool blocking=  CL_TRUE;
	char* hostPtr=  (char*)glMapBufferRange( GL_ARRAY_BUFFER,m_instancingRenderer->getMaxShapeCapacity(),posArraySize, GL_MAP_WRITE_BIT|GL_MAP_READ_BIT );//GL_READ_WRITE);//GL_WRITE_ONLY
		GLint err = glGetError();
    assert(err==GL_NO_ERROR);
	glFinish();

	

#if 1



	//do some stuff using the OpenCL buffer

	bool useCpu = false;
	if (useCpu)
	{
		

		float* posBuffer = (float*)hostPtr;
		
		for (int i=0;i<numParticles;i++)
		{
			posBuffer[i*4+1] += 0.1;
		}
	}
	else
	{
		cl_int ciErrNum;
		if (!m_data->m_clPositionBuffer)
		{
			m_data->m_clPositionBuffer = clCreateBuffer(m_data->m_clContext, CL_MEM_READ_WRITE,
				posArraySize, 0, &ciErrNum);

			clFinish(m_data->m_clQueue);
			oclCHECKERROR(ciErrNum, CL_SUCCESS);
			ciErrNum = clEnqueueWriteBuffer (	m_data->m_clQueue,m_data->m_clPositionBuffer,
 				blocking,0,posArraySize,hostPtr,0,0,0
			);
			clFinish(m_data->m_clQueue);
		}
	
		if (1)
		{
			btBufferInfoCL bInfo[] = { 
				btBufferInfoCL( m_data->m_velocitiesGPU->getBufferCL(), true ),
				btBufferInfoCL( m_data->m_clPositionBuffer)
			};
			
			btLauncherCL launcher(m_data->m_clQueue, m_data->m_updatePositionsKernel );

			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(btBufferInfoCL) );
			launcher.setConst( numParticles);

			launcher.launch1D( numParticles);
			clFinish(m_data->m_clQueue);
	
		}

		if (1)
		{
			ciErrNum = clEnqueueReadBuffer (	m_data->m_clQueue,
				m_data->m_clPositionBuffer,
	 			blocking,
 				0,
 				posArraySize,
 			hostPtr,0,0,0);

			//clReleaseMemObject(clBuffer);
			clFinish(m_data->m_clQueue);

			
		}
	}
	
#endif

	glUnmapBuffer( GL_ARRAY_BUFFER);
	glFlush();

	/*
	int numParticles = NUM_PARTICLES_X*NUM_PARTICLES_Y*NUM_PARTICLES_Z;
	for (int objectIndex=0;objectIndex<numParticles;objectIndex++)
	{
		float pos[4]={0,0,0,0};
		float orn[4]={0,0,0,1};

//		m_instancingRenderer->writeSingleInstanceTransformToGPU(pos,orn,i);
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_instancingRenderer->getInternalData()->m_vbo);
			glFlush();

			char* orgBase =  (char*)glMapBuffer( GL_ARRAY_BUFFER,GL_READ_WRITE);
			//btGraphicsInstance* gfxObj = m_graphicsInstances[k];
			int totalNumInstances= numParticles;
	

			int POSITION_BUFFER_SIZE = (totalNumInstances*sizeof(float)*4);

			char* base = orgBase;
			int capInBytes = m_instancingRenderer->getMaxShapeCapacity();

			float* positions = (float*)(base+capInBytes);
			float* orientations = (float*)(base+capInBytes+ POSITION_BUFFER_SIZE);

			positions[objectIndex*4+1] += 0.1f;
			glUnmapBuffer( GL_ARRAY_BUFFER);
			glFlush();
		}
	}
	*/

	
}

//	m_data->m_positionOffsetInBytes = demo.m_maxShapeBufferCapacity/4;


hasCL = findOpenCL_NVIDIA()
	
if (hasCL) then
	
	project "bullet2_gpu_demo_opengl2_NVIDIA"

	initOpenCL_NVIDIA()

	language "C++"
			
	kind "ConsoleApp"
	
	targetdir "../../../../bin"

		includedirs {
              "..",
              "../../../../bullet2"
    }
	

	links { 
		"BulletSoftBody",
		"BulletDynamics",
		"BulletCollision",
		"LinearMath"
	}
	

	initOpenGL()
	initGlew()

	files {
			"../GpuDemo.cpp",
			"../GpuDemo.h",
			"../btGpuDynamicsWorld.cpp",
			"../btGpuDynamicsWorld.h",
			"../btCpuDynamicsWorld.cpp",
			"../btCpuDynamicsWorld.h",
			"../btGpuIntegrateTransforms.cpp",
			"../btGpuIntegrateTransforms.h",
			"../main_opengl2.cpp",
			
			"../../../../opencl/opengl_interop/btOpenCLGLInteropBuffer.cpp",
			"../../../../opencl/opengl_interop/btOpenCLGLInteropBuffer.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/btGpuSapBroadphase.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/btGpuSapBroadphase.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/ConvexHullContact.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/ConvexHullContact.h",
			"../../../../opencl/broadphase_benchmark/btPrefixScanCL.cpp",
			"../../../../opencl/broadphase_benchmark/btPrefixScanCL.h",
			"../../../../opencl/broadphase_benchmark/btRadixSort32CL.cpp",
			"../../../../opencl/broadphase_benchmark/btRadixSort32CL.h",
			"../../../../opencl/broadphase_benchmark/btFillCL.cpp",
			"../../../../opencl/broadphase_benchmark/btFillCL.h",
			"../../../../opencl/broadphase_benchmark/btBoundSearchCL.cpp",
			"../../../../opencl/broadphase_benchmark/btBoundSearchCL.h",
			"../../../../opencl/gpu_rigidbody_pipeline/btConvexUtility.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline/btConvexUtility.h",
			"../../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h",
			"../../../../dynamics/basic_demo/ConvexHeightFieldShape.cpp",
			"../../../../dynamics/basic_demo/ConvexHeightFieldShape.h",
			"../../../../dynamics/basic_demo/Stubs/ChNarrowphase.cpp",
			"../../../../dynamics/basic_demo/Stubs/Solver.cpp",
			"../../../../opencl/broadphase_benchmark/findPairsOpenCL.cpp",
			"../../../../opencl/broadphase_benchmark/findPairsOpenCL.h",
			"../../../../opencl/broadphase_benchmark/btGridBroadphaseCL.cpp",
			"../../../../opencl/broadphase_benchmark/btGridBroadphaseCL.h",
			"../../../../opencl/3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.cpp",
			"../../../../opencl/3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.h",
			"../../../../opencl/3dGridBroadphase/Shared/btGpu3DGridBroadphase.cpp",
			"../../../../opencl/3dGridBroadphase/Shared/btGpu3DGridBroadphase.h",

			"../../../../opencl/basic_initialize/btOpenCLUtils.cpp",
			"../../../../opencl/basic_initialize/btOpenCLUtils.h",
			"../../../../opencl/basic_initialize/btOpenCLInclude.h",
			
			
			
			"../../../DemosCommon/GL_ShapeDrawer.cpp",
			"../../../DemosCommon/GL_ShapeDrawer.h",
			"../../../DemosCommon/OpenGL2Renderer.cpp",
			"../../../DemosCommon/OpenGL2Renderer.h",
			"../../../../rendering/rendertest/GLPrimitiveRenderer.cpp",
			"../../../../rendering/rendertest/GLPrimitiveRenderer.h",
			"../../../../rendering/rendertest/Win32OpenGLWindow.cpp",
			"../../../../rendering/rendertest/Win32OpenGLWindow.h",
			"../../../../rendering/rendertest/Win32Window.cpp",
			"../../../../rendering/rendertest/Win32Window.h",
			"../../../../rendering/rendertest/LoadShader.cpp",
			"../../../../rendering/rendertest/LoadShader.h",
											
	}

	project "bullet2_gpu_demo_opengl3core_NVIDIA"

	initOpenCL_NVIDIA()

	language "C++"
			
	kind "ConsoleApp"
	
	targetdir "../../../../bin"

		includedirs {
              "..",
              "../../../../bullet2"
    }
	

	links { 
		"BulletSoftBody",
		"BulletDynamics",
		"BulletCollision",
		"LinearMath"
	}
	

	initOpenGL()
	initGlew()

	files {
			"../GpuDemo.cpp",
			"../GpuDemo.h",
			"../btGpuDynamicsWorld.cpp",
			"../btGpuDynamicsWorld.h",
			"../btCpuDynamicsWorld.cpp",
			"../btCpuDynamicsWorld.h",
			"../btGpuIntegrateTransforms.cpp",
			"../btGpuIntegrateTransforms.h",
			
			"../main_opengl3core.cpp",
	
			"../../../../opencl/opengl_interop/btOpenCLGLInteropBuffer.cpp",
			"../../../../opencl/opengl_interop/btOpenCLGLInteropBuffer.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/CLPhysicsDemo.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/btGpuSapBroadphase.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/btGpuSapBroadphase.h",
			"../../../../opencl/gpu_rigidbody_pipeline2/ConvexHullContact.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline2/ConvexHullContact.h",
			"../../../../opencl/broadphase_benchmark/btPrefixScanCL.cpp",
			"../../../../opencl/broadphase_benchmark/btPrefixScanCL.h",
			"../../../../opencl/broadphase_benchmark/btRadixSort32CL.cpp",
			"../../../../opencl/broadphase_benchmark/btRadixSort32CL.h",
			"../../../../opencl/broadphase_benchmark/btFillCL.cpp",
			"../../../../opencl/broadphase_benchmark/btFillCL.h",
			"../../../../opencl/broadphase_benchmark/btBoundSearchCL.cpp",
			"../../../../opencl/broadphase_benchmark/btBoundSearchCL.h",
			"../../../../opencl/gpu_rigidbody_pipeline/btConvexUtility.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline/btConvexUtility.h",
			"../../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.cpp",
			"../../../../opencl/gpu_rigidbody_pipeline/btGpuNarrowPhaseAndSolver.h",
			"../../../../dynamics/basic_demo/ConvexHeightFieldShape.cpp",
			"../../../../dynamics/basic_demo/ConvexHeightFieldShape.h",
			"../../../../dynamics/basic_demo/Stubs/ChNarrowphase.cpp",
			"../../../../dynamics/basic_demo/Stubs/Solver.cpp",
			"../../../../opencl/broadphase_benchmark/findPairsOpenCL.cpp",
			"../../../../opencl/broadphase_benchmark/findPairsOpenCL.h",
			"../../../../opencl/broadphase_benchmark/btGridBroadphaseCL.cpp",
			"../../../../opencl/broadphase_benchmark/btGridBroadphaseCL.h",
			"../../../../opencl/3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.cpp",
			"../../../../opencl/3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.h",
			"../../../../opencl/3dGridBroadphase/Shared/btGpu3DGridBroadphase.cpp",
			"../../../../opencl/3dGridBroadphase/Shared/btGpu3DGridBroadphase.h",

			"../../../../opencl/basic_initialize/btOpenCLUtils.cpp",
			"../../../../opencl/basic_initialize/btOpenCLUtils.h",
			"../../../../opencl/basic_initialize/btOpenCLInclude.h",


			"../../../DemosCommon/GL_ShapeDrawer.cpp",
			"../../../DemosCommon/GL_ShapeDrawer.h",
			"../../../DemosCommon/OpenGL3CoreRenderer.cpp",
			"../../../DemosCommon/OpenGL3CoreRenderer.h",
			"../../../../rendering/rendertest/GLInstancingRenderer.cpp",
			"../../../../rendering/rendertest/GLInstancingRenderer.h",
			"../../../../rendering/rendertest/GLPrimitiveRenderer.cpp",
			"../../../../rendering/rendertest/GLPrimitiveRenderer.h",
			"../../../../rendering/rendertest/Win32OpenGLWindow.cpp",
			"../../../../rendering/rendertest/Win32OpenGLWindow.h",
			"../../../../rendering/rendertest/Win32Window.cpp",
			"../../../../rendering/rendertest/Win32Window.h",
			"../../../../rendering/rendertest/LoadShader.cpp",
			"../../../../rendering/rendertest/LoadShader.h",
											
	}
end
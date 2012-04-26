	
	hasCL = findOpenCL_NVIDIA()
	
	if (hasCL) then

		project "OpenCL_broadphase_benchmark_NVIDIA"

		initOpenCL_NVIDIA()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

		initOpenGL()
		initGlut()
		initGlew()

			includedirs {
		"../../../rendering/BulletMath",
		"../../primitives",
		"../../../bullet2"
		}
		
		files {
			"../main.cpp",
			"../findPairsOpenCL.cpp",
			"../findPairsOpenCL.h",
			"../btGridBroadphaseCL.cpp",
			"../btGridBroadphaseCL.h",
			"../../broadphase_benchmark/btPrefixScanCL.cpp",
			"../../broadphase_benchmark/btPrefixScanCL.h",
			"../../broadphase_benchmark/btRadixSort32CL.cpp",
			"../../broadphase_benchmark/btRadixSort32CL.h",
			"../../broadphase_benchmark/btFillCL.cpp",
			"../../broadphase_benchmark/btFillCL.h",
			"../../3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.cpp",
			"../../3dGridBroadphase/Shared/bt3dGridBroadphaseOCL.h",
			"../../3dGridBroadphase/Shared/btGpu3DGridBroadphase.cpp",
			"../../3dGridBroadphase/Shared/btGpu3DGridBroadphase.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h",
			"../../../bullet2/BulletCollision/BroadphaseCollision/btBroadphaseProxy.cpp",
			"../../../bullet2/BulletCollision/BroadphaseCollision/btOverlappingPairCache.cpp",
			"../../../bullet2/BulletCollision/BroadphaseCollision/btSimpleBroadphase.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h",
			"../../opengl_interop/btOpenCLGLInteropBuffer.cpp",
			"../../opengl_interop/btOpenCLGLInteropBuffer.h",
			"../../opengl_interop/btStopwatch.cpp",
			"../../opengl_interop/btStopwatch.h"
		}
		
	end
	
	hasCL = findOpenCL_NVIDIA()
	
	if (hasCL) then

		project "OpenGL_rendertest_NVIDIA"

		initOpenCL_NVIDIA()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"


		initOpenGL()
		initGlew()

		includedirs {
		"../../primitives",
		"../../../bullet2"
		}
		
			
		
		files {
			"../main.cpp",
			"../renderscene.cpp",
			"../renderscene.h",
			"../fontstash.cpp",
			"../fontstash.h",
			"../stb_image_write.h",
			"../stb_truetype.h",
			"../../gpu_rigidbody_pipeline2/GLInstancingRenderer.cpp",
			"../../gpu_rigidbody_pipeline2/GLInstancingRenderer.h",
			"../../gpu_rigidbody_pipeline2/Win32OpenGLRenderManager.cpp",
			"../../gpu_rigidbody_pipeline2/Win32OpenGLRenderManager.h",	
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h",
			"../../opengl_interop/btOpenCLGLInteropBuffer.cpp",
			"../../opengl_interop/btOpenCLGLInteropBuffer.h",
			"../../opengl_interop/btStopwatch.cpp",
			"../../opengl_interop/btStopwatch.h"
		}
		
	end

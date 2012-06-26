	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenGL_rendertest_Apple"

		initOpenCL_Apple()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"


		initOpenGL()
		initGlew()

		includedirs {
		"../../primitives",
		"../../../bullet2"
		}
		
			
		links {"BulletFileLoader"}
		links { "Cocoa.framework" }
		
		files {
			"../main.cpp",
			"../renderscene.cpp",
			"../renderscene.h",
			"../MacOpenGLWindow.h",
			"../MacOpenGLWindow.mm",
			"../../gpu_rigidbody_pipeline2/GLInstancingRenderer.cpp",
			"../../gpu_rigidbody_pipeline2/GLInstancingRenderer.h",
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h",
			--"../../opengl_interop/btOpenCLGLInteropBuffer.cpp",
			--"../../opengl_interop/btOpenCLGLInteropBuffer.h",
			"../../opengl_interop/btStopwatch.cpp",
			"../../opengl_interop/btStopwatch.h"
		}
		
	end

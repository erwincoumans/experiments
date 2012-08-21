
	project "OpenGL_bullet2_pipeline_AMD"


		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"


		initOpenGL()
		initGlew()

		
		includedirs {
		"../../../rendering/rendertest",
		"../../primitives",
		"../../../bullet2"
		}
		
			
		links {
		"BulletDynamics",
		"BulletCollision",
		"LinearMath"}

		
		files {

			"../main.cpp",
			"../physics_func.cpp",
			"../physics_func.h",
			"../../../rendering/rendertest/GLInstancingRenderer.cpp",
			"../../../rendering/rendertest/GLInstancingRenderer.h",
			"../../../rendering/rendertest/Win32OpenGLRenderManager.cpp",
			"../../../rendering/rendertest/Win32OpenGLRenderManager.h",	
			"../../../rendering/rendertest/LoadShader.cpp",
			"../../../rendering/rendertest/LoadShader.h",
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h"

	}
		


	project "OpenGL_bullet2_pipeline_Apple"


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

		links { "Cocoa.framework" }
		
		files {

			"../main.cpp",
			"../physics_func.cpp",
			"../physics_func.h",
			"../../../rendering/rendertest/GLInstancingRenderer.cpp",
			"../../../rendering/rendertest/GLInstancingRenderer.h",
      "../../../rendering/rendertest/MacOpenGLWindow.mm",
      "../../../rendering/rendertest/MacOpenGLWindow.h",
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h"

	}
		

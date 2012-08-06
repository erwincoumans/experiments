	
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
			"../GLInstancingRenderer.cpp",
			"../GLInstancingRenderer.h",
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h"
		}
		
	end

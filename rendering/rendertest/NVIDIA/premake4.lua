	
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
			"../../../bullet2",
		 	"../../Gwen",
		}
		
		links {
			"gwen"
		}
		
		files {
			"../main.cpp",
			"../renderscene.cpp",
			"../renderscene.h",
			"../GLInstancingRenderer.cpp",
			"../GLInstancingRenderer.h",
			"../Win32OpenGLRenderManager.cpp",
			"../Win32OpenGLRenderManager.h",	
			"../GLPrimitiveRenderer.h",
			"../GLPrimitiveRenderer.cpp",
			"../LoadShader.cpp",
			"../LoadShader.h",
			"../gwenWindow.cpp",
			"../gwenWindow.h",
            "../GwenOpenGL3CoreRenderer.h",
			"../../OpenGLTrueTypeFont/fontstash.cpp",
			"../../OpenGLTrueTypeFont/fontstash.h",
			"../../OpenGLTrueTypeFont/opengl_fontstashcallbacks.cpp",
 			"../../OpenGLTrueTypeFont/opengl_fontstashcallbacks.h",
			"../../../bullet2/LinearMath/btConvexHullComputer.cpp",
			"../../../bullet2/LinearMath/btConvexHullComputer.h",
			"../../../bullet2/LinearMath/btSerializer.cpp",
			"../../../bullet2/LinearMath/btSerializer.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h"
		}
		
	end

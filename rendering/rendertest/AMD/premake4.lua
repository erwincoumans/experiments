	
	hasCL = findOpenCL_AMD()
	
	if (hasCL) then

		project "OpenGL_rendertest_AMD"

		initOpenCL_AMD()
	
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
			"../GLPrimitiveRenderer.h",
			"../GLPrimitiveRenderer.cpp",
			"../LoadShader.cpp",
			"../LoadShader.h",
			"../gwenWindow.cpp",
			"../gwenWindow.h",
			"../TwFonts.cpp",
			"../TwFonts.h",
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

		if os.is("Windows") then 
			files{  
				"../Win32OpenGLWindow.cpp",
                        	"../Win32OpenGLWindow.h",
                        	"../Win32Window.cpp",
                        	"../Win32Window.h",
			}
		end
		if os.is("Linux") then
			files {
				"../X11OpenGLWindow.cpp",
				"../X11OpenGLWindows.h"
			}
		end
	end

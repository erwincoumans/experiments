	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenGL_TrueTypeFont_Apple"

		initOpenCL_Apple()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"


		initOpenGL()
		initGlew()

		includedirs {
		"../../rendertest",
		"../../primitives",
		"../../../bullet2"
		}
		
			
		links {"BulletFileLoader"}
		links { "Cocoa.framework" }
		
		files {
			"../main.cpp",
			"../../rendertest/MacOpenGLWindow.h",
			"../../rendertest/MacOpenGLWindow.mm",
			"../../rendertest/LoadShader.cpp",
			"../../rendertest/LoadShader.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../bullet2/LinearMath/btQuickprof.cpp",
			"../../../bullet2/LinearMath/btQuickprof.h" ,
			"../fontstash.cpp",
      "../fontstash.h",
      "../opengl_fontstashcallbacks.cpp",
      "../opengl_fontstashcallbacks.h",
			"../stb_image_write.h",
      "../stb_truetype.h",
			}
		
	end

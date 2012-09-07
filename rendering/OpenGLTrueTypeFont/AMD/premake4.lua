	
	hasCL = findOpenCL_AMD()
	
	if (hasCL) then

		project "OpenGL_TrueTypeFont_AMD"

		initOpenCL_AMD()
	
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
		
			
---		links {"BulletFileLoader"}
		
		files {
			"../main.cpp",
			"../../rendertest/Win32OpenGLWindow.cpp",
			"../../rendertest/Win32OpenGLWindow.h",
			"../../rendertest/Win32Window.cpp",
			"../../rendertest/Win32Window.h",
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

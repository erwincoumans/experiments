	if os.is("Windows") then

	project "Gwen_OpenGLTest_Windows"
		
	kind "ConsoleApp"
	flags {"Unicode"}
	
	defines { "GWEN_COMPILE_STATIC" , "_HAS_EXCEPTIONS=0", "_STATIC_CPPLIB" }
	
	targetdir "../../../../bin"
	
	includedirs 
	{
	
		"../..","..","../../../rendertest","../../../../bullet2"
	}

	initOpenGL()
	initGlew()
			
	links {
		"gwen",
	}
	
	
	files {
		"../../../rendertest/Win32OpenGLWindow.cpp",
		"../../../rendertest/Win32OpenGLWindow.h",
		"../../../rendertest/Win32Window.cpp",
		"../../../rendertest/Win32Window.h",
		"../../../rendertest/TwFonts.cpp",
		"../../../rendertest/TwFonts.h",
		"../../../rendertest/LoadShader.cpp",
		"../../../rendertest/LoadShader.h",
		"../../../rendertest/GLPrimitiveRenderer.cpp",
		"../../../rendertest/GLPrimitiveRenderer.h",				
		"../../../rendertest/GwenOpenGL3CoreRenderer.h",
		"../../../OpenGLTrueTypeFont/fontstash.cpp",
		"../../../OpenGLTrueTypeFont/fontstash.h",
		"../../../OpenGLTrueTypeFont/opengl_fontstashcallbacks.cpp",
 		"../../../OpenGLTrueTypeFont/opengl_fontstashcallbacks.h",
		"../../../../bullet2/LinearMath/btConvexHullComputer.cpp",
		"../../../../bullet2/LinearMath/btConvexHullComputer.h",
		"../../../../bullet2/LinearMath/btSerializer.cpp",
		"../../../../bullet2/LinearMath/btSerializer.h",
		"../../../../bullet2/LinearMath/btAlignedAllocator.cpp",
		"../../../../bullet2/LinearMath/btQuickprof.cpp",
		"../../../../bullet2/LinearMath/btQuickprof.h",
		"../**.cpp",
		"../**.h",
	}
	end


	project "Gwen_OpenGLTest"
		
	kind "ConsoleApp"
	flags {"Unicode"}
	
	defines { "GWEN_COMPILE_STATIC" , "_HAS_EXCEPTIONS=0", "_STATIC_CPPLIB" }
	
	targetdir "../../../bin"
	
	includedirs 
	{
		"../../GlutGlewWindows",
		".."
	}
	libdirs {"../../GlutGlewWindows"}

	links {
		"gwen",
		"glew32",
		"opengl32"
	}
	
	
	files {
		"**.cpp",
		"**.h",
	}
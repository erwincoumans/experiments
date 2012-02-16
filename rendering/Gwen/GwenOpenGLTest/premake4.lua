	if os.is("Windows") then

	project "Gwen_OpenGLTest"
		
	kind "ConsoleApp"
	flags {"Unicode"}
	
	defines { "GWEN_COMPILE_STATIC" , "_HAS_EXCEPTIONS=0", "_STATIC_CPPLIB" }
	
	targetdir "../../../bin"
	
	includedirs 
	{
	
		".."
	}

	initOpenGL()
	initGlew()
			
	links {
		"gwen",
	}
	
	
	files {
		"**.cpp",
		"**.h",
	}
	end


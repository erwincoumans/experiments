	project "GLES2_Simple_Texture2D_Angle"
		
	kind "ConsoleApp"
		
	includedirs 
	{
		"."
	}
	libdirs {"../GlutGlewWindows"}

	links {
		"libEGL",
		"libGLESv2",
	}
	
	
	files {
		"Simple_Texture2D.c",
		"esShader.cpp",
		"esShapes.c",
		"esTransform.c",
		"esUtil.cpp",
		"esUtil_TGA.cpp",
		"esUtil_win32.c"		
	}
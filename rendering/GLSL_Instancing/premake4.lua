	project "GLSL_instancing"
		
	kind "ConsoleApp"
	targetdir "../../bin"
	
	includedirs 
	{
		"../GlutGlewWindows",
		"../BulletMath"
	}
	libdirs {"../GlutGlewWindows"}

	links {
		"glut32",
		"glew32",
		"opengl32"
	}
	
	
	files {
		"main.cpp"
	}
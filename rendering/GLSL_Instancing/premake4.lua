	if os.is("Windows") then

	project "GLSL_instancing"
		
	kind "ConsoleApp"
	targetdir "../../bin"
	
	includedirs 
	{
		"../BulletMath"
	}

	initOpenGL()
	initGlut()
	initGlew()
		
	files {
		"main.cpp"
	}
	end


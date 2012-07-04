	if os.is("Windows") then

	project "GLES2_Simple_Texture2D_Angle"
		
	kind "WindowedApp"
		
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
		"Simple_Texture2D.cpp",
		"esShader.cpp",
		"esShapes.cpp",
		"esTransform.cpp",
		"esUtil.cpp",
		"esUtil_TGA.cpp",
		"esUtil_win32.cpp"		
	}
	
	
	project "GLES2_BulletBlendReader_Angle"
		
	kind "WindowedApp"
		
	includedirs 
	{
		".",
		"../BlenderSerialize",
		"../../bullet2",
		"../../jpeglib"
	}
	libdirs {"../GlutGlewWindows"}

	links {
		"libEGL",
		"libGLESv2",
		"BulletCollision",
		"LinearMath",
		"BulletDynamics",
		"jpeglib"
	}
	
	
	files {
		"btTransformUtil.cpp",
		"BulletBlendReaderNew.cpp",
		"esShader.cpp",
		"esUtil.cpp",
		"esUtil_win32.cpp",
		"OolongReadBlend.cpp",
		"Simple_Texture2DSetupAndRenderFrame.cpp",
		"SimpleGLSL2.cpp",
		
		"../BlenderSerialize/bBlenderFile.cpp",
		"../BlenderSerialize/bChunk.cpp",
		"../BlenderSerialize/bDNA.cpp",
		"../BlenderSerialize/bFile.cpp",
		"../BlenderSerialize/bMain.cpp",
		"../BlenderSerialize/dna249.cpp",
		"../BlenderSerialize/dna249-64bit.cpp",


	}
	end


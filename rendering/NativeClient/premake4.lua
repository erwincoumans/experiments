	project "NativeClientTumbler"
		
	kind "ConsoleApp"
	
	
	 includedirs { "."	}
	 includedirs { 
	 								"../../bullet2"	, 
	 										"../../jpeglib",
	 								"../BlenderSerialize"
	 						 }

	--libdirs {}

	links {
		"ppapi_gles2",
		"ppapi",
		"ppapi_cpp",
		"ppruntime",
		"bullet2",
		"jpeglib"
	}
	
	
	files {
			"../OpenGLES2Angle/Simple_Texture2DSetupAndRenderFrame.cpp",
				"../OpenGLES2Angle/BulletBlendReaderNew.cpp",
					"../BlenderSerialize/bBlenderFile.cpp",
					"../OpenGLES2Angle/OolongReadBlend.cpp",
		"../BlenderSerialize/bChunk.cpp",
		"../BlenderSerialize/bDNA.cpp",
		"../BlenderSerialize/bFile.cpp",
		"../BlenderSerialize/bMain.cpp",
		"../BlenderSerialize/dna249.cpp",
		"../BlenderSerialize/dna249-64bit.cpp",
			--"BulletDemo.cpp",
			--"BulletDemo.h",
				"cube.cc",
        "opengl_context.cc",
        "scripting_bridge.cc",
        "shader_util.cc",
        "transforms.cc",
        "tumbler.cc",
        "tumbler_module.cc",
        "btTransformUtil.cpp",
				"btTransformUtil.h"

	}
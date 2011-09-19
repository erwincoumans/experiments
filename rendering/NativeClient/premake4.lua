	project "NativeClientTumbler"
		
	kind "ConsoleApp"
	
	
	 includedirs { "."	}
	 includedirs { "../../bullet2"	}

	--libdirs {}

	links {
		"ppapi_gles2",
		"ppapi",
		"ppapi_cpp",
		"ppruntime",
		"bullet2"
	}
	
	
	files {
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
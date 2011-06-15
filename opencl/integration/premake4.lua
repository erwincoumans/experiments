	
	hasCL = findOpenCL()
	
	if (hasCL) then

		project "OpenCL_integration"

		initOpenCL()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		libdirs {"../../rendering/GlutGlewWindows"}

		links {
			"glut32",
			"glew32",
			"opengl32"
		}
			includedirs {
		"../../rendering/GlutGlewWindows",
		"../../rendering/BulletMath"
		}
		
		files {
			"main.cpp",
			"../basic_initialize/btOpenCLUtils.cpp",
			"../basic_initialize/btOpenCLUtils.h",
			"../opengl_interop/btOpenCLGLInteropBuffer.cpp",
			"../opengl_interop/btOpenCLGLInteropBuffer.h",
			"../opengl_interop/btStopwatch.cpp",
			"../opengl_interop/btStopwatch.h"
		}
		
	end
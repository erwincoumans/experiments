	
		project "corotational_fem"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		libdirs {"../../rendering/GlutGlewWindows"}

		links {
--			"bullet2",
			"glut32",
			"glew32",
			"opengl32"
		}
		
		includedirs {
		".",
		"../../rendering/GlutGlewWindows",
		"../../bullet2",
		"../testbed"
		}
		
		files {
		"application.cpp",
			"OpenTissue/**.h"
		}
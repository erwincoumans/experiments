	
		project "position_based_dynamics"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		libdirs {"../../rendering/GlutGlewWindows"}

		links {
			"bullet2",
			"testbed",
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
		"**.cpp",
		"**.h"
		}
	
		project "corotational_fem"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		libdirs {"../../rendering/GlutGlewWindows"}

		links {
			"gwen",
			"glut32",
			"glew32",
			"opengl32"
		}
		
		includedirs {
		".",
		"../../rendering/GlutGlewWindows",
		"../../rendering/Gwen",
		"../testbed"
		}
		
		files {
			"**.cpp",
			"**.h"
		}
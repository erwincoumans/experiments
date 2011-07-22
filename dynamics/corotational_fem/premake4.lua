	
		project "corotational_fem"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		links {
			"gwen",
		}

		

		includedirs {
		".",
		"../../rendering/Gwen",
		"../testbed"
		}

		
                configuration {"not Windows", "not MaxOSX"}
		links {
                "GL", "GLU", "glut"
                }

                configuration {"Windows"}
		libdirs {"../../rendering/GlutGlewWindows"}
		includedirs {
		"../../rendering/GlutGlewWindows",
		}
		links {
			"glut32",
			"glew32",
			"opengl32"
		}

		configuration {}

		files {
			"**.cpp",
			"**.h"
		}
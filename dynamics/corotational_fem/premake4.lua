	
		project "corotational_fem"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

		links {
			"gwen",
		}

		configuration{}


		

		includedirs {
		".",
		"../../rendering/Gwen",
		"../testbed"
		}

		
                configuration {"not Windows", "not MaxOSX"}
		links {
                "GL", "GLU", "glut","GLEW"
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

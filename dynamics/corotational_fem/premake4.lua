	
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

		configuration {"MacOSX"}
                links { "Carbon.framework","OpenGL.framework","AGL.framework","Glut.framework" }
                configuration {"not Windows", "not MacOSX"}
                links {"GL","GLU","glut"}

		
                configuration {"not Windows", "not MacOSX"}
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

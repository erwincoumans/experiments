	
		project "position_based_dynamics"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

  		includedirs {
                ".",
                "../../bullet2",
                "../testbed"
                }
		

		links { "testbed",
			"bullet2",
		}
		
		configuration { "Windows" }
 		links { "glut32","glew32","opengl32" }
		includedirs{	"../../rendering/GlutGlewWindows"	}
 		libdirs {"../../rendering/GlutGlewWindows"}


		configuration {"MaxOSX"}
 		linkoptions { "-framework Carbon -framework OpenGL -framework AGL -framework Glut" } 
		configuration {"not Windows", "not MacOSX"}
		links {"GL","GLU","glut"}
	
		configuration{}
	
		files {
		"**.cpp",
		"**.h"
		}

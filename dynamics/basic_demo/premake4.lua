if os.is("Windows") then
	
		project "basic_bullet2_demo"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

  		includedirs {
                ".",
                "../../bullet2",
                "../testbed",
                	"../../rendering/Gwen",
                }
		

		links { "testbed",
			"bullet2",
			"gwen"
		}
		

	
		configuration { "Windows" }
 		links { "glut32","glew32","opengl32" }
		includedirs{	"../../rendering/GlutGlewWindows"	}
 		libdirs {"../../rendering/GlutGlewWindows"}


		configuration {"MacOSX"}
 		links { "Carbon.framework","OpenGL.framework","AGL.framework","Glut.framework" } 
		configuration {"not Windows", "not MacOSX"}
		links {"GL","GLU","glut"}
	
		configuration{}
	
		files {
		"**.cpp",
		"**.h"
		}

end

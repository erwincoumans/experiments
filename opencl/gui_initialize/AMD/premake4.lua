	hasCL = findOpenCL_AMD()
	
	if (hasCL) then
		
		project "OpenCL_GUI_Intialize_AMD"

		initOpenCL_AMD()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

 		includedirs {
                "..",
                "../../../rendering/Gwen",
                }

		links {
			"gwen"
		}
		

	
		configuration { "Windows" }
 		links { "glut32","glew32","opengl32" }
		includedirs{	"../../../rendering/GlutGlewWindows"	}
 		libdirs {"../../../rendering/GlutGlewWindows"}


		configuration {"MaxOSX"}
 		linkoptions { "-framework Carbon -framework OpenGL -framework AGL -framework Glut" } 
		configuration {"not Windows", "not MacOSX"}
		links {"GL","GLU","glut"}
	
		configuration{}
	
		files {
		"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h"
		}

		
	end
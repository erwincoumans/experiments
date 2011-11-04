
include "AMD"

	if os.is("Windows") then	
		project "CDTestFramework"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

	configuration {"Windows"}
		defines {"USE_ANTTWEAKBAR"}
	configuration{}
	
	defines { "BAN_OPCODE_AUTOLINK","WIN32","ICE_NO_DLL"}
	
  		includedirs {
                ".",
                "../../bullet2",
                "Opcode",
                "AntTweakBar/include",
                --"../testbed",
                	"../../rendering/Gwen",
                }
		

		links { 
		  "opcode",
			"bullet2",
			"AntTweakBarStatic"
			--"gwen"
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
		"*.cpp",
		"*.h"
		}
end

include "AMD"
--include "Intel"
--include "NVIDIA"


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
		

	
		initOpenGL()
		initGlut()

	
		files {
		"*.cpp",
		"*.h"
		}
end
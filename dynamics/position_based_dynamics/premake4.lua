	
		project "position_based_dynamics"

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
			"BulletSoftBody",
			"BulletDynamics",
			"BulletCollision",
			"LinearMath",
			"gwen"
		}
		

		initOpenGL()
		initGlut()

	
		files {
		"**.cpp",
		"**.h"
		}

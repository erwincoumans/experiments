	
		project "mesh_mesh_test"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

  		includedirs {
                ".",
                "../../bullet2",
                "../testbed",
                }
		

		links { "testbed",
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

	
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

		initOpenGL()
		initGlut()
		initGlew()


		files {
			"**.cpp",
			"**.h"
		}

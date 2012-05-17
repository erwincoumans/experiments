	
		project "exact-ccd"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

--		links {
--		}

		configuration{}

		includedirs {
		".",
		}

		files {
			"**.cpp",
			"**.h"
		}

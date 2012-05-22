
	
		project "exact-ccd"

		flags {"FloatStrict"}
		
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

--		links {
--		}

		includedirs {
		".",
		}

	
		files {
			"**.cpp",
			"**.h"
		}

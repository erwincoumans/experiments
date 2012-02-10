	
		project "profiler_test"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

  		includedirs {
                ".",
                "../../bullet2",
                }

		files {
		"main.cpp",
		"../../bullet2/LinearMath/btQuickprof.cpp"
		}

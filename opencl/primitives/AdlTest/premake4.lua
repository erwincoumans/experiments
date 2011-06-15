	
	hasCL = findOpenCL()
	hasDX11 = findDirectX11()
	
	if (hasCL) then

		project "gpu_research_unit_test"

		initOpenCL()

		if (hasDX11) then
			initDirectX11()
		end
		
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"
		includedirs {"../../../include/gpu_research"}
		
		links {
		"OpenCL"
		}
		
		files {
			"main.cpp",
			"RadixSortBenchmark.h",
			"UnitTests.h"
		}
		
	end
	
	hasCL = findOpenCL()
	
	if (hasCL) then

		project "OpenCL_intialize"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

--		includedirs {"../../../include/gpu_research"}
		
		files {
			"main.cpp",
			"btOpenCLUtils.cpp",
			"btOpenCLUtils.h"
		}
		
	end
	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenCL_intialize_Apple"

		initOpenCL_Apple()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

--		includedirs {"..","../../../../include/gpu_research"}
		
		files {
			"../main.cpp",
			"../btOpenCLUtils.cpp",
			"../btOpenCLUtils.h"
		}
		
	end

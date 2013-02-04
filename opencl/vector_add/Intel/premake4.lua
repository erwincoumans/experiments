	
	hasCL = findOpenCL_Intel()
	
	if (hasCL) then

		project "OpenCL_VectorAdd_Intel"

		initOpenCL_Intel()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

		files {
			"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h"
		}
		
	end
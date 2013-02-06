	
	hasCL = findOpenCL_NVIDIA()
	
	if (hasCL) then

		project "OpenCL_VectorAdd_NVIDIA"

		initOpenCL_NVIDIA()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

		files {
			"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h"
		}
		
	end
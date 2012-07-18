	
	hasCL = findOpenCL_NVIDIA()
	
	if (hasCL) then

		project "OpenCL_compiler_NVIDIA"

		initOpenCL_NVIDIA()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

		includedirs {"..","../../basic_initialize"}
		
		files {
			"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h"
		}
		
	end
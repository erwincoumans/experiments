	
	hasCL = findOpenCL_Intel()
	
	if (hasCL) then

		project "OpenCL_compiler_Intel"

		initOpenCL_Intel()
	
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
	
	hasCL = findOpenCL_AMD()
	
	if (hasCL) then

		project "OpenCL_compiler_AMD"

		initOpenCL_AMD()
	
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
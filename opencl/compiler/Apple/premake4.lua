	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenCL_compiler_Apple"

		initOpenCL_Apple()
	
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

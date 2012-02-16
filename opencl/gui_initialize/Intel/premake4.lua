	hasCL = findOpenCL_Intel()
	
	if (hasCL) then
		
		project "OpenCL_GUI_Intialize_Intel"

		initOpenCL_Intel()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

 		includedirs {
                "..",
                "../../../rendering/Gwen",
                }

		links {
			"gwen"
		}
		

	
		initOpenGL()
		initGlut()
	
		files {
		"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h"
		}

		
	end
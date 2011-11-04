	
	hasCL = findOpenCL_AMD()
	
	if (hasCL) then

		project "OpenCL_SharedLibraryIntialize_AMD"

		initOpenCL_AMD()
	
		language "C++"
				
		kind "SharedLib"
--		targetdir "../../../bin"

--		includedirs {"..","../../../../include/gpu_research"}
		
		files {
			"../main.cpp",
			"../btOpenCLUtils.cpp",
			"../btOpenCLUtils.h"
		}
		
			project "OpenCL_SharedLibraryTest_AMD"
			kind "ConsoleApp"
			links{"OpenCL_SharedLibraryIntialize_AMD"}

			files {
				"main.cpp"
			}

	end
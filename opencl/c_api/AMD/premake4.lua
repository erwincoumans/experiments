	
	hasCL = findOpenCL_AMD()
	
	if (hasCL) then

		project "OpenCL_C_API_Test"

		initOpenCL_AMD()
	
		kind "ConsoleApp"
		targetdir "../../../bin"

		language "C"
		files {
			"../main.c",
		}

		language "C++"
		files {
			"../../basic_initialize/btOpenCLUtils.cpp"
		}
		
	end
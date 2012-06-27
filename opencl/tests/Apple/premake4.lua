	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenCL_tests_Apple"

		initOpenCL_Apple()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

	 	includedirs {
                "../../primitives",
                "../../../bullet2",
                "../../../dynamics/basic_demo"
                }

                links {"BulletFileLoader"}
                links { "Cocoa.framework" }
	
		files {
			"../main.cpp",
			"../../basic_initialize/btOpenCLUtils.cpp",
			"../../basic_initialize/btOpenCLUtils.h",
			"../../../bullet2/LinearMath/btAlignedAllocator.cpp"
		}
		
	end

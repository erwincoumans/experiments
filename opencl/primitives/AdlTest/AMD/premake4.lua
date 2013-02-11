	
	hasCL = findOpenCL_AMD()
	hasDX11 = findDirectX11()
	
	if (hasCL) then

		project "OpenCL_DX11_primitives_test_AMD"

		initOpenCL_AMD()

		if (hasDX11) then
			initDirectX11()
		end
		
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../../bin"
		includedirs {"..","../.."}
		
		links {
		"OpenCL"
		}
		
		files {
			"../main.cpp",
			"../../../basic_initialize/btOpenCLInclude.h",
			"../../../basic_initialize/btOpenCLUtils.cpp",
			"../../../basic_initialize/btOpenCLUtils.h",
			"../../../broadphase_benchmark/btFillCL.cpp",
			"../../../broadphase_benchmark/btFillCL.h",
			"../../../broadphase_benchmark/btBoundSearchCL.cpp",
			"../../../broadphase_benchmark/btBoundSearchCL.h",
			"../../../broadphase_benchmark/btPrefixScanCL.cpp",
			"../../../broadphase_benchmark/btPrefixScanCL.h",
			"../../../broadphase_benchmark/btRadixSort32CL.cpp",
			"../../../broadphase_benchmark/btRadixSort32CL.h",
			"../../../../bullet2/LinearMath/btAlignedAllocator.cpp",
			"../../../../bullet2/LinearMath/btAlignedAllocator.h",
			"../../../../bullet2/LinearMath/btAlignedObjectArray.h",
		}
		
	end
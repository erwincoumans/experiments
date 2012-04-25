	
	hasCL = findOpenCL_AMD()
	hasDX11 = findDirectX11()
	
	if (hasCL) then

		project "OpenCL_radixsort_benchmark_AMD"

		initOpenCL_AMD()

		if (hasDX11) then
			initDirectX11()
		end
		
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../../bin"
		includedirs {"..",projectRootDir .. "bullet2"}
		
		links {
			"OpenCL","bullet2"
		}
		
		files {
			"../../../basic_initialize/btOpenCLUtils.cpp",
			"../../../basic_initialize/btOpenCLUtils.h",
			"../../../broadphase_benchmark/btFillCL.cpp",
			"../../../broadphase_benchmark/btPrefixScanCL.cpp",
			"../../../broadphase_benchmark/btRadixSort32CL.cpp",
			"../test_large_problem_sorting.cpp"
		}
		
	end
	
	hasCL = findOpenCL_Apple()
	
	if (hasCL) then

		project "OpenCL_radixsort_benchmark_Apple"

		initOpenCL_Apple()

		
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../../bin"
		includedirs {"..",projectRootDir .. "bullet2"}
		
		links {
			"bullet2"
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

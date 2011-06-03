solution "0MySolution"

	configurations {"Release", "Debug"}
	configuration "Release"
		flags { "Optimize", "StaticRuntime", "NoRTTI", "NoExceptions"}
	configuration "Debug"
		flags { "Symbols", "StaticRuntime" , "NoRTTI", "NoExceptions"}
	platforms {"x32", "x64"}

	configuration "x64"		
		targetsuffix "_64"
	configuration {"x64", "debug"}
		targetsuffix "_x64_debug"
	configuration {"x64", "release"}
		targetsuffix "_x64"
	configuration {"x32", "debug"}
		targetsuffix "_debug"


	function findOpenCL()
		local amdopenclpath = os.getenv("AMDAPPSDKROOT")
		if (amdopenclpath) then
			return true
		end
		local nvidiaopenclpath = os.getenv("CUDA_PATH")
		if (nvidiaopenclpath) then
			return true
		end
		return false
	end
			
	function initOpenCL()
	-- todo: add Apple and Intel OpenCL environment vars
	-- todo: allow multiple SDKs
	
		configuration {}
		local amdopenclpath = os.getenv("AMDAPPSDKROOT")
		if (amdopenclpath) then
			defines { "ADL_ENABLE_CL" , "CL_PLATFORM_AMD"}
			includedirs {
				"$(AMDAPPSDKROOT)/include"				
			}
			configuration "x32"
				libdirs {"$(AMDAPPSDKROOT)/lib/x86"}
			configuration "x64"
				libdirs {"$(AMDAPPSDKROOT)/lib/x86_64"}
			configuration {}
	
			links {"OpenCL"}
			return true
		end

		configuration {}
		local nvidiaopenclpath = os.getenv("CUDA_PATH")
		if (nvidiaopenclpath) then
			defines { "ADL_ENABLE_CL" , "CL_PLATFORM_NVIDIA"}
			includedirs {
				"$(CUDA_PATH)/include"				
			}
			configuration "x32"
				libdirs {"$(CUDA_PATH)/lib/Win32"}
			configuration "x64"
				libdirs {"$(CUDA_PATH)/lib/x64"}
			configuration {}

			links {"OpenCL"}

			return true
		end

		

		return false
	end


	language "C++"
	location "build"
	targetdir "bin"

	include "../opencl/opengl_interop"
	include "../opencl/basic_initialize"
	include "../rendering/GLSL_Instancing"


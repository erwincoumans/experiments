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


	dofile ("findOpenCL.lua")
	dofile ("findDirectX11.lua")
	
	language "C++"
	location "build"
	targetdir "bin"

--	include "../opencl/opengl_interop"
--	include "../opencl/integration"
--	include "../opencl/primitives/AdlTest"
--	include "../rendering/GLSL_Instancing"
--	include "../opencl/basic_initialize"

	include "../wxwidgets/wxWidgetsGLTest"
	include "../wxwidgets/wxWidgets-2.9.1/wxCMake/wxAll"
	
	
	
	
	
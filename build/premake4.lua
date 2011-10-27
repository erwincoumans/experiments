solution "0MySolution"

	-- Multithreaded compiling
	if _ACTION == "vs2010" or _ACTION=="vs2008" then
		buildoptions { "/MP"  }
	end 
	
	newoption {
    trigger     = "with-nacl",
    description = "Enable Native Client build"
  }
  
	configurations {"Release", "Debug"}
	configuration "Release"
		flags { "Optimize", "StaticRuntime", "NoMinimalRebuild", "FloatFast"}
	configuration "Debug"
		flags { "Symbols", "StaticRuntime" , "NoMinimalRebuild", "NoEditAndContinue" ,"FloatFast"}
		
	platforms {"x32", "x64"}

	configuration "x64"		
		targetsuffix "_64"
	configuration {"x64", "debug"}
		targetsuffix "_x64_debug"
	configuration {"x64", "release"}
		targetsuffix "_x64"
	configuration {"x32", "debug"}
		targetsuffix "_debug"

	configuration{}

if not _OPTIONS["with-nacl"] then
		flags { "NoRTTI", "NoExceptions"}
		defines { "_HAS_EXCEPTIONS=0" }
		targetdir "../bin"
	  location("./" .. _ACTION)

else
--	targetdir "../bin_html"

--remove some default flags when cross-compiling for Native Client
--see also http://industriousone.com/topic/how-remove-usrlib64-x86-builds-cross-compiling
	premake.gcc.platforms.x64.ldflags = string.gsub(premake.gcc.platforms.x64.ldflags, "-L/usr/lib64", "")
	premake.gcc.platforms.x32.ldflags = string.gsub(premake.gcc.platforms.x32.ldflags, "-L/usr/lib32", "")
	
	targetdir "nacl/nginx-1.1.2/html"
	
	location("./nacl")
end


	dofile ("findOpenCL.lua")
	dofile ("findDirectX11.lua")
	
	language "C++"
	

	include "../bullet2"	
	include "../jpeglib"

	

	
if not _OPTIONS["with-nacl"] then

	include "../opencl/opengl_interop"
	include "../opencl/integration"
	include "../opencl/primitives/AdlTest"
	include "../opencl/primitives/benchmark"
	include "../rendering/GLSL_Instancing"
	include "../opencl/basic_initialize"
	include "../opencl/gui_initialize"
	
	
	include "../physics_effects/base_level"
	include "../physics_effects/low_level"
	include "../physics_effects/util"
	include "../physics_effects/sample_api_physics_effects/0_console"
	
	include "../physics_effects/sample_api_physics_effects/1_simple"
	include "../physics_effects/sample_api_physics_effects/2_stable"
	include "../physics_effects/sample_api_physics_effects/3_sleep"
	include "../physics_effects/sample_api_physics_effects/4_motion_type"
	include "../physics_effects/sample_api_physics_effects/5_raycast"
	include "../physics_effects/sample_api_physics_effects/6_joint"

	include "../dynamics/testbed"
	include "../dynamics/position_based_dynamics"

	
	include "../dynamics/corotational_fem"
	--include "../dynamics/nncg_test"

	include "../rendering/Gwen/Gwen"
	include "../rendering/Gwen/GwenOpenGLTest"
	include "../rendering/OpenGLES2Angle"
else
	include "../rendering/NativeClient"	
	
end

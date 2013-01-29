
--                flags { "Symbols", "StaticRuntime" , "NoMinimalRebuild", "NoEditAndContinue" ,"FloatFast"}
 
  solution "0MySolution"

	-- Multithreaded compiling
	if _ACTION == "vs2010" or _ACTION=="vs2008" then
		buildoptions { "/MP"  }
	end 
	
	act = ""
    
    if _ACTION then
        act = _ACTION
    end


	newoption 
	{
    		trigger     = "ios",
    		description = "Enable iOS target (requires xcode4)"
  	}
	
	newoption {
    		trigger     = "with-nacl",
    		description = "Enable Native Client build"
  	}
  
  	newoption {
    		trigger     = "with-pe",
    		description = "Enable Physics Effects"
  	}
  
	configurations {"Release", "Debug"}
	configuration "Release"
		flags { "Optimize", "EnableSSE","StaticRuntime", "NoMinimalRebuild", "FloatFast"}
	configuration "Debug"
		defines {"_DEBUG=1"}
		flags { "Symbols", "StaticRuntime" , "NoMinimalRebuild", "NoEditAndContinue" ,"FloatFast"}
		
	platforms {"x32", "x64"}

	configuration {"x32"}
		targetsuffix ("_" .. act)
	configuration "x64"		
		targetsuffix ("_" .. act .. "_64" )
	configuration {"x64", "debug"}
		targetsuffix ("_" .. act .. "_x64_debug")
	configuration {"x64", "release"}
		targetsuffix ("_" .. act .. "_x64_release" )
	configuration {"x32", "debug"}
		targetsuffix ("_" .. act .. "_debug" )
	
	configuration{}

	postfix=""

	if _ACTION == "xcode4" then
		if _OPTIONS["ios"] then
      			postfix = "ios";
      			xcodebuildsettings
      			{
              		'CODE_SIGN_IDENTITY = "iPhone Developer"',
              		"SDKROOT = iphoneos",
              		'ARCHS = "armv7"',
              		'TARGETED_DEVICE_FAMILY = "1,2"',
              		'VALID_ARCHS = "armv7"',
      			}      
      		else
      			xcodebuildsettings
      			{
              		'ARCHS = "$(ARCHS_STANDARD_32_BIT) $(ARCHS_STANDARD_64_BIT)"',
              		'VALID_ARCHS = "x86_64 i386"',
      			}
    		end
	end





	if not _OPTIONS["with-nacl"] then
		flags { "NoRTTI", "NoExceptions"}
		defines { "_HAS_EXCEPTIONS=0" }
		targetdir "../bin"
	  	location("./" .. act .. postfix)

	else
--	targetdir "../bin_html"
--remove some default flags when cross-compiling for Native Client
--see also http://industriousone.com/topic/how-remove-usrlib64-x86-builds-cross-compiling
		premake.gcc.platforms.x64.ldflags = string.gsub(premake.gcc.platforms.x64.ldflags, "-L/usr/lib64", "")
		premake.gcc.platforms.x32.ldflags = string.gsub(premake.gcc.platforms.x32.ldflags, "-L/usr/lib32", "")
		targetdir "nacl/nginx-1.1.2/html"
		location("./nacl")
	end

	

	projectRootDir = os.getcwd() .. "/../"
	print("Project root directroy: " .. projectRootDir);

	dofile ("findOpenCL.lua")
	dofile ("findDirectX11.lua")
	dofile ("findOpenGLGlewGlut.lua")
	
	language "C++"
	
	if _OPTIONS["ios"] then
		include "../rendering/iOSnew"
	end


	--include "../dynamics/ros"
	include "../bullet2"	
	include "../jpeglib"

	 if not _OPTIONS["ios"] then

	
	--include "../bullet2/Demos/BasicDemo"
	include "../bullet2/Demos/GpuDemo"
	include "../bullet2/Demos/GpuDemo2/Apple"
	include "../bullet2/Demos/OldParticleDemo"

	include "../opencl/gpu_rigidbody_pipeline2"
	
	include "../opencl/basic_initialize"
	include "../opencl/gui_initialize"
	
	include "../rendering/Gwen/Gwen"
	include "../rendering/Gwen/GwenOpenGLTest"
	
	
	--include "../pole"
	
	include "../rendering/OpenGLTrueTypeFont"
	include "../opencl/gui_initialize"
	include "../rendering/rendertest"


	include "../dynamics/position_based_dynamics"
	include "../dynamics/testbed"

	include "../opencl/c_api"
	include "../dynamics/meshmesh"

	 include "../opencl/compiler"

 include "../opencl/gpu_narrowphase_test"

	include "../rendering/OpenGLES2Angle"

		if false then
--	if true then
--if not _OPTIONS["with-nacl"] then


	include "../opencl/gpu_narrowphase_test"
	
	
	include "../opencl/compiler"
	include "../opencl/vector_add"
	include "../opencl/opengl_interop"
	include "../opencl/global_atomics"
--	include "../opencl/integration"

	include "../opencl/benchmark/sort"
--	include "../opencl/primitives/benchmark"
	include "../rendering/GLSL_Instancing"

	include "../opencl/3dGridBroadphase"
	include "../opencl/broadphase_benchmark"
--	include "../opencl/gpu_rigidbody_pipeline"
	
	include "../opencl/tests"
	
	
	include "../dynamics/profiler_test"
	--include "../Lua"
	
	
if _OPTIONS["with-pe"] then
	
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

	include "../opencl/physics_effects_pipeline"
	


end
	include "../opencl/bullet2_pipeline"
--	include "../opencl/bullet3_pipeline"
	
	
	
	include "../dynamics/basic_demo"
--	include "../dynamics/bullet_serialize"

	include "../dynamics/exact-ccd"
	
	include "../dynamics/corotational_fem"
	--include "../dynamics/nncg_test"

	
	include "../rendering/WavefrontObjLoader"

	
	
end
end


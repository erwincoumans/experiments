	if os.is("Windows") then

	project "pe_sample_6_joint"
		
	kind "WindowedApp"
	targetdir "../../../bin"
	includedirs {"../../../physics_effects"}
		
	links {
		"physics_effects_low_level",
		"physics_effects_base_level",
		"physics_effects_util",
		"opengl32"
	}
	
	flags       {"WinMain"}
	
	files {
		"main.cpp",
		"physics_func.cpp",
		"barrel.h",
		"landscape.h",
		"../common/ctrl_func.win32.cpp",
		"../common/perf_func.win32.cpp",
		"../common/render_func.win32.cpp"
	}
	end


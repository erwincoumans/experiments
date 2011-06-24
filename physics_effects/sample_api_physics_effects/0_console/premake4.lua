	project "pe_sample_0_console"
		
	kind "ConsoleApp"
	targetdir "../../../bin"
	includedirs {"../../../physics_effects"}
		
	links {
		"physics_effects_low_level",
		"physics_effects_base_level",
		"physics_effects_util"
	}
	
	files {
		"main.cpp",
		"physics_func.cpp",
		"physics_func.h",
		"../common/perf_func.win32.cpp"		
	}
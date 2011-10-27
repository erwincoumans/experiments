	if os.is("Windows") then

	project "pe_sample_1_simple"
		
	kind "WindowedApp"
	targetdir "../../../bin"
	includedirs {"../../../physics_effects"}
		
	links {
		"physics_effects_low_level",
		"physics_effects_base_level",
		"physics_effects_util",
	}
	
	
	files {
		"main.cpp",
		"physics_func.cpp"
}
	flags {"WinMain"}
	files {
		"../common/ctrl_func.win32.cpp",
		"../common/perf_func.win32.cpp",
		"../common/render_func.win32.cpp"
	}
	links {"opengl32"}
	end


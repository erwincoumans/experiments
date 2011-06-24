	project "physics_effects_low_level"
		
	kind "StaticLib"
	targetdir "../../build/lib"	
	includedirs {
		".."
	}
	files {
		"**.cpp",
		"../../include/physics_effects/low_level/**.h"

	}
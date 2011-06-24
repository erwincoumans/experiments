	project "physics_effects_base_level"
		
	kind "StaticLib"
	targetdir "../../build/lib"	
	includedirs {
		"..",
	}
	files {
		"**.cpp",
		"../../include/physics_effects/base_level/**.h"

	}
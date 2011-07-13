	project "bullet2"
		
	kind "StaticLib"
	targetdir "../build/lib"	
	includedirs {
		".",
	}
	files {
		"**.cpp",
		"**.h"
	}
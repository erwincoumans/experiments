	project "testbed"
		
	kind "StaticLib"
	targetdir "../../build/lib"	
	includedirs {
		".",
		"../../bullet2",
		"../../rendering/GlutGlewWindows"
	}
	files {
		"**.cpp",
		"**.h"
	}
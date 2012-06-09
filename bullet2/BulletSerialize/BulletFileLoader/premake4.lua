	project "BulletFileLoader"
		
	kind "StaticLib"
	targetdir "../../../build/lib"
	includedirs {
		"../.."
	}
	 
	files {
		"**.cpp",
		"**.h"
	}
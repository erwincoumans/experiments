	project "bullet2"
		
	kind "StaticLib"
	targetdir "../build/lib"	
	includedirs {
		"."
	}
	files {
		"BulletSoftBody/**.cpp",
	  "BulletSoftBody/**.h",
		"BulletDynamics/**.cpp",
	  "BulletDynamics/**.h",
		"BulletCollision/**.h",
		"BulletCollision/**.cpp",
		"BulletCollision/**.h",
		"LinearMath/**.h",
		"LinearMath/**.cpp",
		"**.h"
	}
	
	include "BulletSerialize/BulletFileLoader"
	if os.is("Windows") then
	project "AntTweakBarStatic"
		
	kind "StaticLib"
	
	--following defines are already hardcoded in AntTweakBar.h
	--defines {"TW_STATIC", "TW_NO_LIB_PRAGMA"}

	defines { "WIN32"}
		
	 
    
	targetdir "../../../../build/lib"	
	includedirs {
		".","../include"
	}
	files {
		"**.cpp",
		"**.h",
		"**.c",
		"**.rc",
		"**.cur"
	}
	end
	
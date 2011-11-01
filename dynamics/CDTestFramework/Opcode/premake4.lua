	if os.is("Windows") then
	project "opcode"
		
	kind "StaticLib"
	
	defines { "OPCODE_EXPORTS" , "WIN32", "ICE_NO_DLL"}
		
	 pchsource ( "StdAfx.cpp" )
   pchheader ( "StdAfx.h" )
    
	targetdir "../../../build/lib"	
	includedirs {
		".",
	}
	files {
		"**.cpp",
		"**.h"
	}
	end
	
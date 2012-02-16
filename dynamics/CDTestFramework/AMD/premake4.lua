	if os.is("Windows") then	

	hasCL = findOpenCL_AMD()
	
	if (hasCL) then
	
		project "CDTestFramework_AMD"

		initOpenCL_AMD()
	
		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../../bin"

	configuration {"Windows"}
		defines {"USE_ANTTWEAKBAR"}
	configuration{}
	
	defines { "BAN_OPCODE_AUTOLINK","WIN32","ICE_NO_DLL"}
	
  		includedirs {
                "..",
                "../../../bullet2",
                "../Opcode",
                "../AntTweakBar/include",
                --"../../testbed",
                	"../../../rendering/Gwen",
									"../../../opencl/3dGridBroadphase/Shared"
                }
		

		links { 
		  "opcode",
			"bullet2",
			"AntTweakBarStatic",
			"OpenCL_bt3dGridBroadphase_AMD"
			--"gwen"
		}
		

	
		initOpenGL()
		initGlut()

	
		files {
		"../*.cpp",
		"../*.h",
		"../../../opencl/basic_initialize/btOpenCLUtils.*"		
		}
		end
		
end
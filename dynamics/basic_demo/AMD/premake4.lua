if os.is("Windows") then
	
		hasCL = findOpenCL_AMD()
	
		if (hasCL) then
	
		project "basic_bullet2_demo_AMD"

		initOpenCL_AMD()
				
		language "C++"
		
		kind "ConsoleApp"
		targetdir "../../../bin"

  		includedirs {
                "..",
                "../../../bullet2",
                "../../testbed",
                "../../../rendering/Gwen",
                "../../../opencl/basic_initialize",
                "../../../opencl/primitives"
                }
		

		links { "testbed",
			"bullet2",
			"gwen"
		}
		
	
		configuration { "Windows" }
 		links { "glut32","glew32","opengl32" }
		includedirs{	"../../../rendering/GlutGlewWindows"	}
 		libdirs {"../../../rendering/GlutGlewWindows"}


		configuration {"MacOSX"}
 		links { "Carbon.framework","OpenGL.framework","AGL.framework","Glut.framework" } 
		configuration {"not Windows", "not MacOSX"}
		links {"GL","GLU","glut"}
	
		configuration{}
	
		files {
		"../**.cpp",
		"../**.h",
		"../../../opencl/basic_initialize/btOpenCLUtils.cpp",
		"../../../opencl/basic_initialize/btOpenCLUtils.h"
		}

	end
	
end

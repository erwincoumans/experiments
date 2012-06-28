
-- include "AMD"

if os.is("Windows") then
	
		project "basic_bullet2_demo"

		language "C++"
				
		kind "ConsoleApp"
		targetdir "../../bin"

  		includedirs {
                ".",
                "../../bullet2",
                "../testbed",
                	"../../rendering/Gwen",
                }
		

		links { "testbed",
			"LinearMath",
			"BulletCollision",
			"BulletDynamics"
	}
		
		initOpenGL()
		initGlut()
	
		files {
		"BasicDemo.cpp",
		"BasicDemo.h",
		"main.cpp"
		}

end


	function initOpenGL()
		configuration {}
		configuration {"Windows"}
			links {"opengl32"}
		configuration {"MacOSX"}
 			links { "Carbon.framework","OpenGL.framework","AGL.framework"} 
		configuration {"not Windows", "not MacOSX"}
			links {"GL","GLU"}
		configuration{}
	end

	function initGlut()
		configuration {}
		configuration {"Windows"}

			includedirs {
				projectRootDir .. "rendering/GlutGlewWindows"
			}
			libdirs { projectRootDir .. "rendering/GlutGlewWindows"}
		configuration {"Windows", "x32"}
			links {"glut32"}
		configuration {"Windows", "x64"}
			links {"glut64"}
	
		configuration {"MacOSX"}
 			links { "Glut.framework" } 
	
		configuration {"not Windows", "not MacOSX"}
			links {"glut"}
		configuration{}
	end

	function initGlew()
		configuration {}
		configuration {"Windows"}
			defines { "GLEW_STATIC"}
			includedirs {
					projectRootDir .. "rendering/GlutGlewWindows"
			}
			libdirs {	projectRootDir .. "rendering/GlutGlewWindows"}
		configuration {"Windows", "x32"}
			links {"glew32s"}
		configuration {"Windows", "x64"}
			links {"glew64s"}

		configuration{}
	end




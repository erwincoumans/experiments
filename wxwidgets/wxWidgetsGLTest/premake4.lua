

	project "wxWidgetsGLTest"
		
	kind "WindowedApp"
	flags { "WinMain" }
	targetdir "../../bin"
	
	includedirs 
	{
		"../wxWidgets-2.9.1/include/setup",
		"../wxWidgets-2.9.1/include",
		"../../rendering/GlutGlewWindows",
		"../../rendering/BulletMath"
	}
	libdirs {"../../rendering/GlutGlewWindows"}

	configuration {"vs2008", "release"}
	libdirs {"../wxWidgets-2.9.1/lib/vs2008release"}

	configuration {"vs2008", "debug"}
	libdirs {"../wxWidgets-2.9.1/lib/vs2008debug"}

	configuration {}
	
	

	links {
		"wxAll",
		"glut32",
		"glew32",
		"opengl32",
		"Comctl32",
		"Rpcrt4"
	}
	
	
	files {
			"main.cpp",
			"MyFrame.cpp"
	}
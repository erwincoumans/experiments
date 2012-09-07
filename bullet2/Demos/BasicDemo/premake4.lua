	
		project "bullet2_basic_demo_opengl2"

		language "C++"
				
		kind "ConsoleApp"
		
		targetdir "../../../bin"

  		includedirs {
                ".",
                "../../../bullet2"
      }
		

		links { 
			"BulletSoftBody",
			"BulletDynamics",
			"BulletCollision",
			"LinearMath"
		}
		

		initOpenGL()
		initGlew()
	
		files {
				"BasicDemo.cpp",
				"BasicDemo.h",
				"main_opengl2.cpp",
				"../../DemosCommon/GL_ShapeDrawer.cpp",
				"../../DemosCommon/GL_ShapeDrawer.h",
				"../../DemosCommon/OpenGL2Renderer.cpp",
				"../../DemosCommon/OpenGL2Renderer.h",
				"../../../rendering/rendertest/GLPrimitiveRenderer.cpp",
				"../../../rendering/rendertest/GLPrimitiveRenderer.h",
				"../../../rendering/rendertest/Win32OpenGLWindow.cpp",
				"../../../rendering/rendertest/Win32OpenGLWindow.h",
				"../../../rendering/rendertest/Win32Window.cpp",
				"../../../rendering/rendertest/Win32Window.h",
				"../../../rendering/rendertest/LoadShader.cpp",
				"../../../rendering/rendertest/LoadShader.h",
												
		}

		project "bullet2_basic_demo_opengl3core"

		language "C++"
				
		kind "ConsoleApp"
		
		targetdir "../../../bin"

  		includedirs {
                ".",
                "../../../bullet2"
      }
		

		links { 
			"BulletSoftBody",
			"BulletDynamics",
			"BulletCollision",
			"LinearMath"
		}
		

		initOpenGL()
		initGlew()
	
		files {
				"BasicDemo.cpp",
				"BasicDemo.h",
				"main_opengl3core.cpp",
				"../../DemosCommon/GL_ShapeDrawer.cpp",
				"../../DemosCommon/GL_ShapeDrawer.h",
				"../../DemosCommon/OpenGL3CoreRenderer.cpp",
				"../../DemosCommon/OpenGL3CoreRenderer.h",
				"../../../rendering/rendertest/GLInstancingRenderer.cpp",
				"../../../rendering/rendertest/GLInstancingRenderer.h",
				"../../../rendering/rendertest/GLPrimitiveRenderer.cpp",
				"../../../rendering/rendertest/GLPrimitiveRenderer.h",
				"../../../rendering/rendertest/Win32OpenGLWindow.cpp",
				"../../../rendering/rendertest/Win32OpenGLWindow.h",
				"../../../rendering/rendertest/Win32Window.cpp",
				"../../../rendering/rendertest/Win32Window.h",
				"../../../rendering/rendertest/LoadShader.cpp",
				"../../../rendering/rendertest/LoadShader.h",
												
		}

	project "bullet2_basic_demo_gles2"

		language "C++"
				
		--kind "ConsoleApp"
		kind "WindowedApp"
			
		targetdir "../../../bin"

  		includedirs {
                ".",
                "../../../bullet2",
                "../../../rendering/OpenGLES2Angle",
      }
		

	links {
		"libEGL",
		"libGLESv2",
	}
	
		links { 
			"BulletSoftBody",
			"BulletDynamics",
			"BulletCollision",
			"LinearMath"
		}
		

		initOpenGL()
		initGlew()
	
		files {
				"BasicDemo.cpp",
				"BasicDemo.h",
				"main_gles2.cpp",
				"../../DemosCommon/GLES2AngleWindow.cpp",
				"../../DemosCommon/GLES2AngleWindow.h",
				"../../../rendering/rendertest/Win32Window.cpp",
				"../../../rendering/rendertest/Win32Window.h",
				"../../DemosCommon/GLES2Renderer.cpp",
				"../../DemosCommon/GLES2Renderer.h",
				"../../DemosCommon/GLES2ShapeDrawer.cpp",
				"../../DemosCommon/GLES2ShapeDrawer.h",
				"../../../rendering/OpenGLES2Angle/btTransformUtil.cpp",
				"../../../rendering/OpenGLES2Angle/btTransformUtil.h",
		}

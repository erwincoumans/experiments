
hasCL = findOpenCL_AMD()
	
if (hasCL) then
	
	project "bullet2_old_particle_demo_glut_AMD"

	initOpenCL_AMD()
	
	initOpenGL()
	initGlut()

	language "C++"
			
	kind "ConsoleApp"
	
	targetdir "../../../../bin"

		includedirs {
              "..",
              "../../../../bullet2",
    }
	

	links { 
		"BulletSoftBody",
		"BulletDynamics",
		"BulletCollision",
		"LinearMath",
		"gwen",
	}
	

	initOpenGL()
	initGlew()

	files {
			"../btParticlesDemoDynamicsWorld.cpp",
			"../btParticlesDemoDynamicsWorld.h",
			"../main.cpp",
			"../ParticlesDemo.cpp",
			"../ParticlesDemo.h",
			"../shaders.cpp",
			"../shaders.h",
			"../DemoApplication.cpp",
			"../DemoApplication.h",
			"../GL_DialogDynamicsWorld.cpp",
			"../GL_DialogWindow.cpp",
			"../GL_ShapeDrawer.cpp",
			"../GL_Simplex1to4.cpp",
			"../GLDebugDrawer.cpp",
			"../GLDebugFont.cpp",
			"../GlutDemoApplication.cpp",
			"../GlutStuff.cpp",
			"../btOpenCLUtils.cpp"	,
	}
end

--solution "0_Solution"
--use 0 prefix so that the file comes first in the folder, in case you have many files


project "MySkeleton"


configuration {}	
kind "WindowedApp"
--even if it is actually a console app, don't use ConsoleApp for iOS

targetdir "bin"

includedirs {	"Include",
		"../../bullet2",
		"../BlenderSerialize",
             	"../../jpeglib"
	}

if _OPTIONS["ios"] then
                        xcodebuildsettings
                        {
                        'INFOPLIST_FILE = "../../rendering/iOSnew/iOS_OpenGL_test-Info.plist"',
                        'OTHER_LDFLAGS = ("-framework",Foundation,"-framework", CoreFoundation,"-framework",GLKit,"-framework",UIKit,"-framework",OpenGLES,"-framework",QuartzCore, "-framework",CoreGraphics)',
			}
end



links {
"jpeglib",	"BulletDynamics","BulletCollision", "LinearMath"
}

language "C++"

files {
	"**.xib",
	"**.cpp",
	"**.h",
	"**.mm",
	"**.m",
	"../OpenGLES2Angle/OolongReadBlend.cpp",
	"../OpenGLES2Angle/BulletBlendReaderNew.cpp",
	"../OpenGLES2Angle/btTransformUtil.cpp",
	"../BlenderSerialize/bDNA.cpp",
	"../BlenderSerialize/bBlenderFile.cpp",
	"../BlenderSerialize/bChunk.cpp",
	"../BlenderSerialize/bFile.cpp",
	"../BlenderSerialize/bMain.cpp",
	"../BlenderSerialize/dna249.cpp",
	"../BlenderSerialize/dna249-64bit.cpp"

}

configuration "**.xib"
	buildaction "Embed"
configuration{}

configuration "**.png"
   buildaction "Embed"
configuration{}


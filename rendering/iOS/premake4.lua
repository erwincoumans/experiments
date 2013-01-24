solution "0_Solution"
--use 0 prefix so that the file comes first in the folder, in case you have many files



 configurations {"Release", "Debug"}
        configuration "Release"
                flags { "Optimize", "StaticRuntime", "NoMinimalRebuild", "FloatFast"}
        configuration "Debug"
                flags { "Symbols", "StaticRuntime" , "NoMinimalRebuild", "NoEditAndContinue" ,"FloatFast"}


project "MySkeleton"


configuration {}	
kind "WindowedApp"
--even if it is actually a console app, don't use ConsoleApp for iOS

targetdir "bin"

includedirs {"Include"}


xcodebuildsettings
{
"INFOPLIST_FILE = Info.plist",
'CODE_SIGN_IDENTITY = "iPhone Developer"',
'OTHER_LDFLAGS = ("-framework",Foundation,"-framework", CoreFoundation,"-framework",UIKit,"-framework",OpenGLES,"-framework",QuartzCore, "-framework",CoreGraphics)',
"SDKROOT = iphoneos",
'ARCHS = "$(ARCHS_STANDARD_32_BIT)"',
'GCC_VERSION = "com.apple.compilers.llvmgcc42"',
'GCC_THUMB_SUPPORT = NO',
'TARGETED_DEVICE_FAMILY = "1,2"',
'STANDARD_C_PLUS_PLUS_LIBRARY_TYPE = dynamic'
}

--links {
--	"BulletDynamics","BulletCollision", "LinearMath"
--}

language "C++"

files {
	"icon.png",
	"**.cpp",
	"**.h",
	"**.mm",
	"**.m"
}


excludes {
	"Renderer/Resource/StreamingWin32.cpp",
	"Utility/Profiler/prof*.cpp",
	"Renderer/Core/GraphicsDevice/GLES20/EAGLView2.m"
}


configuration "**.png"
   buildaction "Embed"
configuration{}


import os
import sys

### Options ################################
opts = Options()
opts.Add(BoolOption('debug'          , 'Can be set to True to include debugging info, or False otherwise', 1))
opts.Add(BoolOption('useSTL'         , 'Can be used to turn on STL support. False, then STL will not be used. True will include the STL files.', 1))
opts.Add(BoolOption('sharedlibrary'  , 'A shared library will be built if this is set to True', 0))
opts.Add(BoolOption('staticlibrary'  , 'A static library will be built if this is set to True', 0))
opts.Add(BoolOption('program'        , 'The xmltest program will be built if this is set to True', 1))

### Source files and targets ###############
libfiles = Split( "tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp " );
libname = 'tinyxml'
binfiles = libfiles + ["xmltest.cpp"]
progname = 'xmltest'


### Set up the environment #################
env = Environment(options = opts, ENV = os.environ )

# If we are in a MSYS shell, prefer the mingw compiler.
if os.name == 'nt' and env[ "ENV" ].has_key( "MSYSTEM" ):
	if env[ "ENV" ]["MSYSTEM"] == "MINGW32":
		print( "Note: switching to gcc tools. This can be disabled in SConstruct." )
		env.Tool( 'mingw' )

### Option Logic ###########################

## debug
if env['debug'] == True:
    env.AppendUnique(CPPDEFINES = ['DEBUG'])

if env['CC'] == 'gcc':
	# gcc and its variants
	if env['debug'] == True:
		# CXXFLAGS includes the value of CCFLAGS
		env.AppendUnique(CCFLAGS    = ['-Wall', '-Wno-format', '-g'])
		env.AppendUnique(LDFLAGS    = ['-g'])
	else:
	    env.AppendUnique(CCFLAGS    = ['-Wall', '-Wno-unknown-pragmas', '-Wno-format', '-O3'])
elif env['CC'] == "cl":
	# Microsoft Visual Studio
	if env['debug'] == True:
		env.AppendUnique(CCFLAGS    = ['/W3', '/ZI', '/Od'])
		env.AppendUnique(LDFLAGS    = ['/debug'])
	else:
		env.AppendUnique(CCFLAGS    = ['/W3', '/O2'])
else:
	print( "Warning: compiler '" + env['CC'] + "' not in SConfigure. Using default flags." );

## useSTL
if env['useSTL'] == True:
	env.AppendUnique(CPPDEFINES = ['TIXML_USE_STL'])

## shadedLibrary
if env['sharedlibrary'] == True:
	env.SharedLibrary(libname, libfiles)

## staticLibrary
if env['staticlibrary'] == True:
	env.StaticLibrary(libname, libfiles)

## program
if env['program'] == True:
    outname = env.Program(progname, binfiles)
	

### Help ###################################
Help(opts.GenerateHelpText(env))

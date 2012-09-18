# Python program to set the version.
##############################################

import re
import sys

def fileProcess( name, lineFunction ):
	filestream = open( name, 'r' )
	if filestream.closed:
		print( "file " + name + " not open." )
		return

	output = ""
	print( "--- Processing " + name + " ---------" )
	while 1:
		line = filestream.readline()
		if not line: break
		output += lineFunction( line )
	filestream.close()
	
	if not output: return			# basic error checking
	
	print( "Writing file " + name )
	filestream = open( name, "w" );
	filestream.write( output );
	filestream.close()
	
	
def echoInput( line ):
	return line

major = input( "Major: " )
minor = input( "Minor: " )
build = input( "Build: " )

print "Setting dox, makedistlinux, tinyxml.h"
print "Version: " + `major` + "." + `minor` + "." + `build`

#### Write the tinyxml.h ####

def engineRule( line ):

	matchMajor = "const int TIXML_MAJOR_VERSION"
	matchMinor = "const int TIXML_MINOR_VERSION"
	matchBuild = "const int TIXML_PATCH_VERSION"

	if line[0:len(matchMajor)] == matchMajor:
		print "1)tinyxml.h Major found"
		return matchMajor + " = " + `major` + ";\n"

	elif line[0:len(matchMinor)] == matchMinor:
		print "2)tinyxml.h Minor found"
		return matchMinor + " = " + `minor` + ";\n"

	elif line[0:len(matchBuild)] == matchBuild:
		print "3)tinyxml.h Build found"
		return matchBuild + " = " + `build` + ";\n"

	else:
		return line;

fileProcess( "tinyxml.h", engineRule )


#### Write the dox ####

def doxRule( line ):

	match = "PROJECT_NUMBER"

	if line[0:len( match )] == match:
		print "dox project found"
		return "PROJECT_NUMBER = " + `major` + "." + `minor` + "." + `build` + "\n"

	else:
		return line;

fileProcess( "dox", doxRule )


#### Write the makedistlinux #####

# example:
# VERSION=2_4_0
linuxRulePattern=re.compile("^VERSION=\d+_\d+_\d+.*$")

def buildlinuxRule( line ):

	m=linuxRulePattern.match(line)
	if not m: return line

	# if here, matched the line
	print "makedistlinux instance found"
	return "VERSION=%d_%d_%d\n"%(major,minor,build)

fileProcess( "makedistlinux", buildlinuxRule )

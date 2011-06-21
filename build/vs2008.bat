rmdir /s /q vs2008
rmdir /s /q build
premake4 clean

rem premake4 --no-pelibs vs2008
rem premake4 --no-pedemos vs2008
rem premake4 --no-bulletlibs --no-pelibs vs2008
premake4 vs2008

rename build vs2008
pause
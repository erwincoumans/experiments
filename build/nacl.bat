
rem make sure that the NACL_SDK_ROOT environment variable is set

./premake4 --with-nacl gmake
cd gmake

set AR=%NACL_SDK_ROOT%\toolchain\win_x86\bin\nacl-ar.exe
set CC=%NACL_SDK_ROOT%\toolchain\win_x86\bin\nacl-gcc.exe
set CXX=%NACL_SDK_ROOT%\toolchain\win_x86\bin\nacl-g++.exe

set config=release32
make

set config=release64
make

pause

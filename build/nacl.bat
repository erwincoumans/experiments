rem make sure that the NACL_SDK_ROOT environment variable is set
set NACL_ENABLE_PPAPI_DEV=1

premake4 --with-nacl gmake
cd nacl

set AR=%NACL_SDK_ROOT%\%NACL_TARGET_PLATFORM%\toolchain\win_x86_newlib\bin\x86_64-nacl-ar.exe
set CC=%NACL_SDK_ROOT%\%NACL_TARGET_PLATFORM%\toolchain\win_x86_newlib\bin\x86_64-nacl-gcc.exe
set CXX=%NACL_SDK_ROOT%\%NACL_TARGET_PLATFORM%\toolchain\win_x86_newlib\bin\x86_64-nacl-g++.exe

set config=release32
make

set config=release64
make

cd ..



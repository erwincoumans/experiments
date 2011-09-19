
echo Only tested using a Cygwin Bash Shell with make.exe available

export AR=$(NACL_SDK_ROOT)/toolchain/win_x86/bin/nacl-ar.exe
export CC=$(NACL_SDK_ROOT)/toolchain/win_x86/bin/nacl-gcc.exe
export CXX=$(NACL_SDK_ROOT)/toolchain/win_x86/bin/nacl-g++.exe

./premake4 --with-nacl gmake
cd gmake
export config=release32
make

export config=release64
make

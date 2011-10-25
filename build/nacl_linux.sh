
echo Only tested using a Cygwin Bash Shell with make.exe available

NACL_SDK_ROOT=$(dirname $(readlink -f $0))/nacl/nacl_sdk/pepper_15
echo $NACL_SDK_ROOT

export AR=$NACL_SDK_ROOT/toolchain/linux_x86_newlib/bin/i686-nacl-ar
export CC=$NACL_SDK_ROOT/toolchain/linux_x86_newlib/bin/i686-nacl-gcc
export CXX=$NACL_SDK_ROOT/toolchain/linux_x86_newlib/bin/i686-nacl-g++

./premake4_linux --with-nacl gmake
cd nacl
export config=release32
make

export config=release64
make

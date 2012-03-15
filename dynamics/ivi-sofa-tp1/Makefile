CXX := g++
NVCC := nvcc

# Compiling in 32-bits can produce faster CUDA kernels
#MACHINE := 32
#MACHINE := 64

# version
#VERSION = 1
#VERSION = 2
#VERSION = 3
#VERSION = 4
# V5: merged linear algebra kernels
#VERSION = 5
# V6: merged reduction kernels
#VERSION = 6

ifeq ($(shell which $(NVCC)),)
NVCC := /usr/local/cuda/bin/nvcc
endif
CC_VENDOR := $(strip $(shell $(CXX) --version | awk 'NR==1 { print $$2 }'))
CC_VERSION := $(strip $(shell $(CXX) --version | awk 'NR==1 { print $$NF }'))
CC_TARGET := $(strip $(shell $(CXX) -v 2>&1 | awk '$$1=="Target:" { print $$2 }'))

ifeq ($(MACHINE),)
ifeq ($(findstring 64, $(CC_TARGET)),)
MACHINE=32
else
MACHINE=64
endif
endif

NVCC_VERSION := $(strip $(shell $(NVCC) --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

OS := $(shell uname)

CXXFLAGS_COMMON = -O3 -I. -m$(MACHINE) -DNDEBUG
ifeq ($(OS),Linux)
# on Linux we expect Boost to be installed
CXXFLAGS_COMMON += -DSOFA_HAVE_BOOST
endif

ifneq ($(VERSION),)
CXXFLAGS_COMMON += -DVERSION=$(VERSION)
endif

CXXFLAGS = $(CXXFLAGS_COMMON) $(ATOMIC_OPT)
CUDAFLAGS = $(CXXFLAGS_COMMON) -Xptxas -v $(ATOMIC_OPT)

# Using GCC >= 4.3, the best know optimization flags
#ifeq ($(CC_VENDOR),(GCC))
ifeq ($(word 1,$(sort 4.3 $(CC_VERSION))),4.3) # here we use sort to compare version number, the lowest one will be the first word
CXXFLAGS += -march=native -mtune=native -fno-math-errno -funroll-loops -mfpmath=387
else # Default options for Intel Core2 processors
CXXFLAGS += -march=nocona -mtune=nocona -fno-math-errno -funroll-loops -mfpmath=387
endif
#endif

#ifeq ($(CC_VENDOR),(GCC))
ifeq ($(word 1,$(sort 4.5 $(CC_VERSION))),4.5) # here we use sort to compare version number, the lowest one will be the first word
CUDAFLAGS += --compiler-bindir=/usr/bin/gcc-4.4
endif
#endif

#ifneq (,$(findstring 3.1, $(NVCC_VERSION)))
# Disable the ABI by default for CUDA 3.1
CUDAFLAGS += -Xptxas -abi=no
# flags for fastest sm_20 code with CUDA 3.1
CUDAFLAGS += -ftz=true -prec-div=false -prec-sqrt=false
#endif
# support for G80 GPU (GeForce 8x00)
CUDAFLAGS += -gencode=arch=compute_10,code=\"compute_10\" 
# support for G92 GPU (GeForce 9x00)
CUDAFLAGS += -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
# support for GT200M GPU (GeForce GT 3x0)
CUDAFLAGS += -gencode=arch=compute_12,code=\"sm_12,compute_12\" 
# support for GT200 GPU (GeForce GTX 2x0)
CUDAFLAGS += -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
# support for GF100 GPU (GeForce GTX 4x0)
CUDAFLAGS += -gencode=arch=compute_20,code=\"sm_20,compute_20\" 

SUFFIX := $(MACHINE)

ifneq ($(VERSION),)
SUFFIX := $(SUFFIX)-v$(VERSION)
endif

SOURCES := main.cpp glut_methods.cpp kernels.cpp simulation.cpp mapping.cpp render.cpp gui.cpp params.cpp cuda/mycuda.cpp sofa/helper/system/thread/CTime.cpp

CUDA_SOURCES := cuda/mycuda.cu cuda/CudaMergedKernels.cu cuda/CudaMechanicalObject.cu cuda/CudaUniformMass.cu cuda/CudaFixedConstraint.cu cuda/CudaPlaneForceField.cu cuda/CudaSphereForceField.cu cuda/CudaTetrahedronFEMForceField.cu cuda/CudaBarycentricMapping.cu cuda/CudaVisualModel.cu

CPU_SOURCES := cpu/CPUMergedKernels.cpp cpu/CPUMechanicalObject.cpp cpu/CPUUniformMass.cpp cpu/CPUFixedConstraint.cpp cpu/CPUPlaneForceField.cpp cpu/CPUSphereForceField.cpp cpu/CPUTetrahedronFEMForceField.cpp cpu/CPUBarycentricMapping.cpp cpu/CPUVisualModel.cpp

HEADERS := $(wildcard *.h mesh/*.h) $(wildcard sofa/defaulttype/*.h sofa/helper/*.h sofa/helper/system/*.h)
CUDA_HEADERS := $(wildcard cuda/*.h)
CPU_HEADERS := $(wildcard cpu/*.h)

OBJECTS_CUDA := $(SOURCES:%.cpp=tmp/cuda_%_$(SUFFIX).o) $(CUDA_SOURCES:%.cu=tmp/cuda_%_cu$(SUFFIX).o)

OBJECTS_CPU := $(SOURCES:%.cpp=tmp/cpu_%_$(SUFFIX).o) $(CPU_SOURCES:%.cpp=tmp/cpu_%_$(SUFFIX).o)

LIB :=

ifeq ($(OS),Darwin)
CXXFLAGS += -Imac/include
LIB += -Lmac/lib -lGLEW
LIB += -framework OpenGL -framework GLUT
LIB += -lpng12 -lz
else
ifneq ($(wildcard linux/lib$(MACHINE)/libGLEW.a),)
LIB += linux/lib$(MACHINE)/libGLEW.a
CXXFLAGS_COMMON += -Ilinux/include
else
LIB += -lGLEW
endif
LIB += -L/usr/X11R6/lib -lglut -lGLU -lGL -lXext -lX11 -lm
LIB += -L/usr/lib$(MACHINE) -lpng -lz
endif

CUDADIR := $(abspath $(dir $(shell which $(NVCC)))../)
ifneq ($(wildcard $(CUDADIR)/lib$(MACHINE)),)
CUDALIBDIR := $(CUDADIR)/lib$(MACHINE)
else
CUDALIBDIR := $(CUDADIR)/lib
endif

LIB_CUDA := -L$(CUDALIBDIR) -lcudart $(LIB)

EXE := 

DEST_CUDA = gpugems-cuda-fem-$(SUFFIX)$(EXE)
DEST_CPU = gpugems-cpu-fem-$(SUFFIX)$(EXE)

.PHONY: all cpu cuda clean

all: cpu

cpu: $(DEST_CPU)

cuda: $(DEST_CUDA)

clean:
	rm -f $(DEST) $(OBJECTS_CUDA) $(OBJECTS_CPU)

$(DEST_CPU): $(OBJECTS_CPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS_CPU) $(LIB)

$(DEST_CUDA): $(OBJECTS_CUDA)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS_CUDA) $(LIB_CUDA)
ifeq ($(OS),Darwin)
	install_name_tool -change @rpath/libcudart.dylib $(CUDALIBDIR)/libcudart.dylib $@
endif

tmp/cpu_%_$(SUFFIX).o: %.cpp $(HEADERS) $(CPU_HEADERS)
	@mkdir -p $(@D)
	$(CXX) -c -DSOFA_DEVICE_CPU $(CXXFLAGS) -o $@ $<

tmp/cuda_%_$(SUFFIX).o: %.cpp $(HEADERS) $(CUDA_HEADERS)
	@mkdir -p $(@D)
	$(CXX) -c -DSOFA_DEVICE_CUDA $(CXXFLAGS) -o $@ $<

tmp/cuda_%_cu$(SUFFIX).o: %.cu $(CUDA_HEADERS)
	@mkdir -p $(@D)
	$(NVCC) -c -DSOFA_DEVICE_CUDA $(CUDAFLAGS) -o $@ $<

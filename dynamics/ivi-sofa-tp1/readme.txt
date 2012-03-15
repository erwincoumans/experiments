             GPU Computing Gems II
   Interactive FEM simulation using CUDA
               Sample code & Demo

Jeremie Allard, Hadrien Courtecuisse, Francois Faure
(c) INRIA 2010
The raptor mesh was created by Jeremy "NoBrainNoBrain" Ringard, used with permission.

STATUS
------

This version should be fully functional and support Linux, Mac and Windows.
The only feature that is missing and might be included later is a double-precision version (kernels are present but not used).
There is currently only one example mesh, in 3 sizes up to 19k elements.
No redistribution license is specified for the source code yet, so please do not redistribute this version.

USAGE
-----

Precompiled binaries are provided for some systems.
If it is not available see below for compilation instructions.
Several versions are available :
- gpugems-cuda-fem-32 / gpugems-cuda-fem-64 : CUDA version in 32-bits or 64-bits
- gpugems-cpu-fem-32 / gpugems-cpu-fem-64 : CPU version in 32-bits or 64-bits

The demo application can be launched with the following parameters :

  --device=num : select the CUDA device (alternatively the CUDA_DEVICE environment variable can be used)
  --bench=iter : run a benchmark of iter timesteps (1000 by default)
  --reorder / --noreorder : control reordering of mesh vertices and tetrahedra
  --novbo : disable vertex buffer object usage to transmit data between CUDA and OpenGL. Use this if there is an issue with your driver (on Mac it might give better performances)
  filename : use another file for the FEM mesh (data/raptor.mesh by default), in Netgen format
  other filenames : load surface meshes to be mapped to the FEM mesh (data/raptor-skin.obj and data/raptor-misc.obj by default), in Wavefront OBJ format

In the data directory, meshes of different sizes are available.

Once the simulation is loaded, the mouse actions can be used :
- clic and draw with the LEFT BUTTON : rotate the camera or apply a force to a particle
- clic and draw with the RIGHT BUTTON : translate the camera or apply a strong force to a particle
- clic and draw with the MIDDLE BUTTON : zoom the camera or apply a constraint to a particle
The following keys are available :
- SPACE : start / stop the simulation
- DEL / SUPPR : reset the simulation
- ENTER : enter / leave fullscreen mode
- ESC : exit the application ( or return to normal mode if in fullscreen )
- LEFT / RIGHT / UP / DOWN / PAGE UP / PAGE DOWN : translate and zoom the camera
- F1 : show / hide help messages
- F2 : show / hide statistics
- F3 : change display mode between surface mesh, FEM mesh, or none
- F4 : display FEM particles
- F5 : display debug info (velocities and forces)

PREREQUISITES
-------------

The CUDA Toolkit needs to be installed and configured (nvcc should be in the PATH).
We recommend at least version 3.1

The following external libraries are used: GLUT, GLEW
On Mac and Windows, a version is included in the mac and win32 directories.
On Linux, please use the packages from your distribution (don't forget the devel packages).


COMPILATION
------------

On Mac and Linux, simply run make.
Look at the top of the Makefile for customizations.
In particular, we default to compiling in 32-bits mode even on 64-bits systems.
This can be changed using the MACHINE variable: make MACHINE=64

On Windows, using Visual Studio 2008 simply double-clic on gpugems-fem.sln and select the appropriate configuration.


SOURCE CODE
-----------

All CUDA kernels are in the cuda directory. In particular, the FEM kernels are in cuda/CudaTetrahedronFEMForceField.cu, and the optimized merged kernels ar in cuda/merged_kernels.cu. Only files within this directory use CUDA directly, the others uses a simple wrapper API defined in mycuda.h. This allows to provide simple traces and error checking, and ease the use of different compilers from the CPU and CUDA codes (such as the Intel compiler, or new versions of gcc or Visual that are not yet supported by CUDA).

A CPU version of all kernels is available in the cpu directory, and used to compile the gpugems-cpu-fem* version of the application.

The sofa directory contains classes providing basic functionalities (fixed-size vectors and matrices, timers), using code from the SOFA physics framework (http://www.sofa-framework.org).
To ease the interface between CPU and GPU codes, we use a sofa::helper::vector container that is modeled after std::vector but which can automatically provide both a CPU and GPU version and automatically keep them synchronized. This is based on deviceRead / deviceWrite and hostRead / hostWrite methods that provide read or write access to the CPU of GPU buffer. Internally, then rely on a MemoryManager template parameter, which implement memory allocation and data transfers using a particular GPU API (the CUDA version is in cuda/CudaMemoryManager.h).

Finally, the application code is contained in the following files :

kernels.cpp : intialization of compute kernels
simulation.cpp : simulation methods, including mesh loading, internal data init, CG solver, implicit time integration, and main animate method
mapping.cpp : attach the visual surface mesh to the motion of the FEM tetrahedral mesh.
render.cpp : opengl rendering
gui.cpp : camera, picking, and displayed text
glut_methods.cpp : GLUT-based window setup and callbacks
params.cpp : simulation parameters (alone so that then can be changed quickly)
main.cpp : main method

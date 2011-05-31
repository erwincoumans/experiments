stringify.py is a small python script 
that converts a file into a header file 
that can be included in a C/C++ program as a string.

It is mainly useful to embed GLSL,OpenCL/HLSL kernels in a C++ program, 
instead of loading them from disk.

Usage: 

stringify.py <original_file> <variablename> > <outfile>

Example: see stringifykernels.bat

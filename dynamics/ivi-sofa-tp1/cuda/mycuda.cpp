/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "mycuda.h"

// namespace sofa
// {
// namespace gpu
// {
// namespace cuda
// {

extern "C"
{
MycudaVerboseLevel mycudaVerboseLevel = LOG_ERR;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_INFO;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_TRACE;
}

void mycudaPrivateInit(int /*device*/)
{
    const char* verbose = getenv("CUDA_VERBOSE");
    if (verbose && *verbose)
        mycudaVerboseLevel = (MycudaVerboseLevel) atoi(verbose);
}

void mycudaLogError(const char* err, const char* src)
{
    std::cerr << "CUDA error: "<< err <<" returned from "<< src <<".\n";
    exit(1);
}

int myprintf(const char* fmt, ...)
{
	va_list args;
	va_start( args, fmt );
	int r = vfprintf( stderr, fmt, args );
	va_end( args );
	return r;
}

const char* mygetenv(const char* name)
{
    return getenv(name);
}

// } // namespace cuda
// } // namespace gpu
// } // namespace sofa

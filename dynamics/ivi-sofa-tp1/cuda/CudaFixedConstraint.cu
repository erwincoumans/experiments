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
#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

extern "C"
{
void CudaFixedConstraint3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE
void CudaFixedConstraint3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void CudaFixedConstraint1t_projectResponseIndexed_kernel(int size, const int* indices, real* dx)
{
    int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
	dx[indices[index]] = 0.0f;
}

template<class real>
__global__ void CudaFixedConstraint3t_projectResponseIndexed_kernel(int size, const int* indices, CudaVec3<real>* dx)
{
    int index = fastmul(blockIdx.x,BSIZE)+threadIdx.x;
    if (index < size)
	dx[indices[index]] = CudaVec3<real>::make(0.0f,0.0f,0.0f);
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaFixedConstraint3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaFixedConstraint3t_projectResponseIndexed_kernel<float><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<float>*)dx);
}

#ifdef SOFA_GPU_CUDA_DOUBLE
void CudaFixedConstraint3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaFixedConstraint3t_projectResponseIndexed_kernel<double><<< grid, threads >>>(size, (const int*)indices, (CudaVec3<double>*)dx);
}
#endif // SOFA_GPU_CUDA_DOUBLE

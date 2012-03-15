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

template<class real>
class GPUSphere
{
public:
    real center_x, center_y, center_z;
    real velocity_x, velocity_y, velocity_z;
    real radius;
    real stiffness;
    real damping;
};

typedef GPUSphere<float> GPUSphere3f;
typedef GPUSphere<double> GPUSphere3d;

extern "C"
{
void CudaSphereForceField3f_addForce(unsigned int size, GPUSphere3f* sphere, float* penetration, void* f, const void* x, const void* v);
void CudaSphereForceField3f_addDForce(unsigned int size, GPUSphere3f* sphere, const float* penetration, void* f, const void* dx); //, const void* dfdx);

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaSphereForceField3d_addForce(unsigned int size, GPUSphere3d* sphere, double* penetration, void* f, const void* x, const void* v);
void CudaSphereForceField3d_addDForce(unsigned int size, GPUSphere3d* sphere, const double* penetration, void* f, const void* dx); //, const void* dfdx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real>
__global__ void CudaSphereForceField3t_addForce_kernel(int size, GPUSphere<real> sphere, real* penetration, real* f, const real* x, const real* v)
{
    // TODO: force computation
}

template<class real>
__global__ void CudaSphereForceField3t_addDForce_kernel(int size, GPUSphere<real> sphere, const real* penetration, real* df, const real* dx)
{
    // TODO: dforce computation
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaSphereForceField3f_addForce(unsigned int size, GPUSphere3f* sphere, float* penetration, void* f, const void* x, const void* v)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaSphereForceField3t_addForce_kernel<float><<< grid, threads >>>(size, *sphere, penetration, (float*)f, (const float*)x, (const float*)v);
}

void CudaSphereForceField3f_addDForce(unsigned int size, GPUSphere3f* sphere, const float* penetration, void* df, const void* dx) //, const void* dfdx)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaSphereForceField3t_addDForce_kernel<float><<< grid, threads >>>(size, *sphere, penetration, (float*)df, (const float*)dx);
}

#ifdef SOFA_GPU_CUDA_DOUBLE

void CudaSphereForceField3d_addForce(unsigned int size, GPUSphere3d* sphere, double* penetration, void* f, const void* x, const void* v)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaSphereForceField3t_addForce_kernel<double><<< grid, threads >>>(size, *sphere, penetration, (double*)f, (const double*)x, (const double*)v);
}

void CudaSphereForceField3d_addDForce(unsigned int size, GPUSphere3d* sphere, const double* penetration, void* df, const void* dx) //, const void* dfdx)
{
	dim3 threads(BSIZE,1);
	dim3 grid((size+BSIZE-1)/BSIZE,1);
	CudaSphereForceField3t_addDForce_kernel<double><<< grid, threads >>>(size, *sphere, penetration, (double*)df, (const double*)dx);
}

#endif // SOFA_GPU_CUDA_DOUBLE

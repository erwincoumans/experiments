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
void CudaTetraMapper3f_apply(unsigned int size, const void* map_i, const void* map_f, void* out, const void* in);
}

//////////////////////
// GPU-side methods //
//////////////////////

template<class TIn>
__global__ void CudaTetraMapper3f_apply_kernel(unsigned int size, const int4* map_i, const float4* map_f, float* out, const TIn* in)
{
    const int index0 = fastmul(blockIdx.x,BSIZE);
    const int index1 = threadIdx.x;
    const int index = index0 + index1;

    __shared__ float temp[BSIZE*3];

    if (index < size)
    {
        int4 in_i = map_i[index];
        float4 in_f = map_f[index];
        CudaVec3<float> res;

        res  = CudaVec3<float>::make(in[in_i.x]) * in_f.x;
        res += CudaVec3<float>::make(in[in_i.y]) * in_f.y;
        res += CudaVec3<float>::make(in[in_i.z]) * in_f.z;
        res += CudaVec3<float>::make(in[in_i.w]) * in_f.w;
        
        const int index3 = fastmul(index1,3);
        
        temp[index3  ] = res.x;
        temp[index3+1] = res.y;
        temp[index3+2] = res.z;
    }
    __syncthreads();

    out += fastmul(index0,3);
    out[index1        ] = temp[index1        ];
    out[index1+  BSIZE] = temp[index1+  BSIZE];
    out[index1+2*BSIZE] = temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaTetraMapper3f_apply(unsigned int size, const void* map_i, const void* map_f, void* out, const void* in)
{
    dim3 threads(BSIZE,1);
    dim3 grid((size+BSIZE-1)/BSIZE,1);
    CudaTetraMapper3f_apply_kernel<CudaVec3<float> ><<< grid, threads >>>(size, (const int4*)map_i, (const float4*)map_f, (float*)out, (const CudaVec3<float>*)in);
}

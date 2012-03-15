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
void CudaVisualModel3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x);
void CudaVisualModel3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x);

void CudaVisualModel3f_calcTNormalsAndTangents(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, void *ftangents, const void* x, const void* tc);
void CudaVisualModel3f_calcVNormalsAndTangents(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, void* vtangents, const void* fnormals, const void* ftangents, const void* x, const void* tc);
}

//////////////////////
// GPU-side methods //
//////////////////////

template<typename real, class TIn>
__global__ void CudaVisualModel3t_calcTNormals_kernel(int nbElem, const int* elems, real* fnormals, const TIn* x)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;
    int index3 = fastmul(index1,3);
    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    __shared__  union {
	int itemp[3*BSIZE];
	real rtemp[3*BSIZE];
    } s;

    s.itemp[index1] = elems[iext];
    s.itemp[index1+BSIZE] = elems[iext+BSIZE];
    s.itemp[index1+2*BSIZE] = elems[iext+2*BSIZE];
    
    __syncthreads();
    
    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        CudaVec3<real> A = x[s.itemp[index3+0]];
        CudaVec3<real> B = x[s.itemp[index3+1]];
        CudaVec3<real> C = x[s.itemp[index3+2]];
        B -= A;
        C -= A;
        N = cross(B,C);
        N *= invnorm(N);
    }

    if (sizeof(real) != sizeof(int)) __syncthreads();

    s.rtemp[index3+0] = N.x;
    s.rtemp[index3+1] = N.y;
    s.rtemp[index3+2] = N.z;

    __syncthreads();

    fnormals[iext] = s.rtemp[index1];
    fnormals[iext+BSIZE] = s.rtemp[index1+BSIZE];
    fnormals[iext+2*BSIZE] = s.rtemp[index1+2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModel3t_calcVNormals_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* velems, real* vnormals, const TIn* fnormals)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = fastmul(index1,3); //3*index1;

    __shared__  real temp[3*BSIZE];

    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    CudaVec3<real> n = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=fastmul(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
    {
	for (int s = 0;s < nbElemPerVertex; s++)
	{
	    int i = *velems -1;
	    velems+=BSIZE;
	    if (i != -1)
            n += fnormals[i];
	}
	real invn = invnorm(n);
	if (invn < 100000.0)
	    n *= invn;
    }

    temp[index3  ] = n.x;
    temp[index3+1] = n.y;
    temp[index3+2] = n.z;

    __syncthreads();

    vnormals[iext        ] = temp[index1        ];
    vnormals[iext+  BSIZE] = temp[index1+  BSIZE];
    vnormals[iext+2*BSIZE] = temp[index1+2*BSIZE];
}



template<typename real, class TIn>
__global__ void CudaVisualModel3t_calcTNormalsAndTangents_kernel(int nbElem, const int* elems, real* fnormals, real* ftangents, const TIn* x, const real* tc)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index = index0+index1;
    int index3 = fastmul(index1,3);
    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    __shared__  union {
	int itemp[3*BSIZE];
	real rtemp[3*BSIZE];
    } s;

    s.itemp[index1] = elems[iext];
    s.itemp[index1+BSIZE] = elems[iext+BSIZE];
    s.itemp[index1+2*BSIZE] = elems[iext+2*BSIZE];
    
    __syncthreads();
    
    CudaVec3<real> N = CudaVec3<real>::make(0,0,0);
    CudaVec3<real> T = CudaVec3<real>::make(0,0,0);
    if (index < nbElem)
    {
        CudaVec3<real> A = x[s.itemp[index3+0]];
        CudaVec3<real> B = x[s.itemp[index3+1]];
        CudaVec3<real> C = x[s.itemp[index3+2]];
        B -= A;
        C -= A;
        N = cross(B,C);
        N *= invnorm(N);
        real Au = tc[(s.itemp[index3+0]<<1)];
        real Bu = tc[(s.itemp[index3+1]<<1)];
        real Cu = tc[(s.itemp[index3+2]<<1)];
        Bu -= Au;
        Cu -= Au;
        T = B * Cu - C * Bu;
        real invT = invnorm(T);
        if (invT < 1000000.0f)
            T *= invT;
    }

    if (sizeof(real) != sizeof(int)) __syncthreads();

    s.rtemp[index3+0] = N.x;
    s.rtemp[index3+1] = N.y;
    s.rtemp[index3+2] = N.z;

    __syncthreads();

    fnormals[iext] = s.rtemp[index1];
    fnormals[iext+BSIZE] = s.rtemp[index1+BSIZE];
    fnormals[iext+2*BSIZE] = s.rtemp[index1+2*BSIZE];

    __syncthreads();

    s.rtemp[index3+0] = T.x;
    s.rtemp[index3+1] = T.y;
    s.rtemp[index3+2] = T.z;

    __syncthreads();

    ftangents[iext] = s.rtemp[index1];
    ftangents[iext+BSIZE] = s.rtemp[index1+BSIZE];
    ftangents[iext+2*BSIZE] = s.rtemp[index1+2*BSIZE];
}

template<typename real, class TIn>
__global__ void CudaVisualModel3t_calcVNormalsAndTangents_kernel(int nbVertex, unsigned int nbElemPerVertex, const int* velems, real* vnormals, real* vtangents, const TIn* fnormals, const TIn* ftangents)
{
    int index0 = fastmul(blockIdx.x,BSIZE); //blockDim.x;
    int index1 = threadIdx.x;
    int index3 = fastmul(index1,3); //3*index1;

    __shared__  real temp[3*BSIZE];

    int iext = fastmul(blockIdx.x,BSIZE*3)+index1; //index0*3+index1;

    CudaVec3<real> n = CudaVec3<real>::make(0.0f,0.0f,0.0f);
    CudaVec3<real> t = CudaVec3<real>::make(0.0f,0.0f,0.0f);

    velems+=fastmul(index0,nbElemPerVertex)+index1;

    if (index0+index1 < nbVertex)
    {
        for (int s = 0;s < nbElemPerVertex; s++)
        {
            int i = *velems -1;
            velems+=BSIZE;
            if (i != -1)
            {
                n += fnormals[i];
                t += ftangents[i];
            }
        }
        t = cross(n,t);
        real invn = invnorm(n);
        if (invn < 1000000.0)
            n *= invn;
        real invt = invnorm(t);
        if (invn < 1000000.0)
            t *= invt;
    }

    temp[index3  ] = n.x;
    temp[index3+1] = n.y;
    temp[index3+2] = n.z;

    __syncthreads();

    vnormals[iext        ] = temp[index1        ];
    vnormals[iext+  BSIZE] = temp[index1+  BSIZE];
    vnormals[iext+2*BSIZE] = temp[index1+2*BSIZE];

    __syncthreads();

    temp[index3  ] = t.x;
    temp[index3+1] = t.y;
    temp[index3+2] = t.z;

    __syncthreads();

    vtangents[iext        ] = temp[index1        ];
    vtangents[iext+  BSIZE] = temp[index1+  BSIZE];
    vtangents[iext+2*BSIZE] = temp[index1+2*BSIZE];
}

//////////////////////
// CPU-side methods //
//////////////////////

void CudaVisualModel3f_calcTNormals(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, const void* x)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    CudaVisualModel3t_calcTNormals_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (float*)fnormals, (const CudaVec3<float>*)x);
}

void CudaVisualModel3f_calcVNormals(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, const void* fnormals, const void* x)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    CudaVisualModel3t_calcVNormals_kernel<float, CudaVec3<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (float*)vnormals, (const CudaVec3<float>*)fnormals);
}


void CudaVisualModel3f_calcTNormalsAndTangents(unsigned int nbElem, unsigned int nbVertex, const void* elems, void* fnormals, void *ftangents, const void* x, const void* tc)
{
    dim3 threads1(BSIZE,1);
    dim3 grid1((nbElem+BSIZE-1)/BSIZE,1);
    CudaVisualModel3t_calcTNormalsAndTangents_kernel<float, CudaVec3<float> ><<< grid1, threads1 >>>(nbElem, (const int*)elems, (float*)fnormals, (float*)ftangents, (const CudaVec3<float>*)x, (const float*)tc);
}

void CudaVisualModel3f_calcVNormalsAndTangents(unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const void* velems, void* vnormals, void* vtangents, const void* fnormals, const void* ftangents, const void* x, const void* tc)
{
    dim3 threads2(BSIZE,1);
    dim3 grid2((nbVertex+BSIZE-1)/BSIZE,1);
    CudaVisualModel3t_calcVNormalsAndTangents_kernel<float, CudaVec3<float> ><<< grid2, threads2 >>>(nbVertex, nbElemPerVertex, (const int*)velems, (float*)vnormals, (float*)vtangents, (const CudaVec3<float>*)fnormals, (const CudaVec3<float>*)ftangents);
}

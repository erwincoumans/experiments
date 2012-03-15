#include "CudaCommon.h"
#include "CudaMath.h"
#include "cuda.h"

template<class real>
class GPUPlane
{
public:
    //CudaVec3<real> normal;
    real normal_x, normal_y, normal_z;
    real d;
    real stiffness;
    //real damping;
};

typedef GPUPlane<float> GPUPlane3f;

extern "C"
{
int CudaMergedKernels3f_cgDeltaTmpSize(unsigned int size);
void CudaMergedKernels3f_cgDelta(bool first, unsigned int size, float* delta, float alpha,
                                 void* r, void* a, const void* q, const void* d,
                                 void* tmp, float* cputmp);
int CudaMergedKernels3f_cgDot3TmpSize(unsigned int size);
void CudaMergedKernels3f_cgDot3(unsigned int size, float* dot3,
                                const void* r, const void* q, const void* d,
                                void* tmp, float* cputmp);
void CudaMergedKernels3f_cgDot3First(unsigned int size, float* dot3,
                                     const void* b, const void* q,
                                     void* tmp, float* cputmp);
void CudaMergedKernels3f_cgOp3(unsigned int size, float alpha, float beta,
                               void* r, void* a, void* d, const void* q);
void CudaMergedKernels3f_cgOp3First(unsigned int size, float alpha, float beta,
                                    void* r, void* a, void* d, const void* q, const void* b);

}

//////////////////////
// GPU-side methods //
//////////////////////

template<class real, bool first, int blockSize>
__global__ void MergedKernels_cgDelta_kernel(unsigned int n, real alpha, real* r, real* a, const real* q, const real* d, real* tmp)
{
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = gridDim.x*(blockSize);
    float sum = 0;
    for (unsigned int i = blockIdx.x*(blockSize) + tid; i < n; i += gridSize)
    {
        real di = d[i];
        real qi = q[i];
        real ai = (first) ? 0 : a[i];
        real ri = (first) ? di : r[i];
        ai += di * alpha;
        a[i] = ai;
        ri -= qi * alpha;
        r[i] = ri;
        sum += ri * ri;
    }

    __shared__ real sdata[blockSize];
    volatile float* smem = sdata;
    smem[tid] = sum;
#define SYNC __syncthreads
    if (blockSize >= 512)
    {
        SYNC();
        if (tid < 256) { smem[tid] += smem[tid + 256]; }
    }
    if (blockSize >= 256)
    {
        SYNC();
        if (tid < 128) { smem[tid] += smem[tid + 128]; } SYNC();
    }
    if (blockSize >= 128)
    {
        SYNC();
        if (tid < 64) { smem[tid] += smem[tid + 64]; }
    }
    SYNC();
    if (tid < 32) {
        if (blockSize >= 64)
            smem[tid] += smem[tid + 32];
        if (blockSize >= 32)
            smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) tmp[blockIdx.x] = smem[0];
#undef SYNC
}

template<class real, bool first, int blockSize>
__global__ void MergedKernels_cgDot3_kernel(unsigned int n, const real* r, const real* q, const real* d, real* tmp)
{
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = gridDim.x*(blockSize);
    float dot_dq = 0;
    float dot_rq = 0;
    float dot_qq = 0;
    for (unsigned int i = blockIdx.x*(blockSize) + tid; i < n; i += gridSize)
    {
        real qi = q[i];
        real di = d[i];
        real ri = (first) ? d[i] : r[i];
        dot_qq += qi*qi;
        dot_dq += di*qi;
        dot_rq += ri*qi;
    }

    __shared__ real sdata[3*blockSize];
    volatile float* smem = sdata;
    unsigned int tid3 = fastmul(tid,3);
    smem[tid3+0] = dot_dq;
    smem[tid3+1] = dot_rq;
    smem[tid3+2] = dot_qq;
#define SYNC __syncthreads
    if (blockSize == 128)
    {
        SYNC();
        if (tid < 120)
            smem[tid] += smem[tid + 120] + smem[tid + 2*120];
        if (tid < 8*3)
            smem[tid] += smem[tid + 3*120];
        SYNC();
        if (tid < 60)
            smem[tid] += smem[tid + 60];
        SYNC();
        if (tid < 30)
            smem[tid] += smem[tid + 30];
        if (tid < 15)
            smem[tid] += smem[tid + 15];
        if (tid < 6)
            smem[tid] += smem[tid + 9];
        if (tid < 3) {
            float sum = smem[tid] + smem[tid + 3] + smem[tid + 6];
            tmp[fastmul(blockIdx.x,3) + tid] = sum;
        }
    }
    else if (blockSize == 64)
    {
        SYNC();
        if (tid < 60)
            smem[tid] += smem[tid + 60] + smem[tid + 2*60];
        if (tid < 4*3)
            smem[tid] += smem[tid + 3*60];
        SYNC();
        if (tid < 30)
            smem[tid] += smem[tid + 30];
        if (tid < 15)
            smem[tid] += smem[tid + 15];
        if (tid < 6)
            smem[tid] += smem[tid + 9];
        if (tid < 3) {
            float sum = smem[tid] + smem[tid + 3] + smem[tid + 6];
            tmp[fastmul(blockIdx.x,3) + tid] = sum;
        }
    }
    else if (blockSize == 32)
    {
        SYNC();
        if (tid < 30)
            smem[tid] += smem[tid + 30] + smem[tid + 2*30];
        if (tid < 2*3)
            smem[tid] += smem[tid + 3*30];
        SYNC();
        if (tid < 15)
            smem[tid] += smem[tid + 15];
        if (tid < 6)
            smem[tid] += smem[tid + 9];
        if (tid < 3) {
            float sum = smem[tid] + smem[tid + 3] + smem[tid + 6];
            tmp[fastmul(blockIdx.x,3) + tid] = sum;
        }
    }
#undef SYNC
}

template<class real, int blockSize>
__global__ void MergedKernels_cgOp3_kernel(unsigned int n, float alpha, float beta,
                                           float* r, float* a, float* d, const float* q)
{
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = gridDim.x*(blockSize);
    for (unsigned int i = blockIdx.x*(blockSize) + tid; i < n; i += gridSize)
    {
        real qi = q[i];
        real di = /* (first) ? b[i] : */ d[i];
        real ai = /* (first) ? 0 : */ a[i];
        real ri = /* (first) ? di : */ r[i];
        ai += di * alpha;
        a[i] = ai;
        ri -= qi * alpha;
        r[i] = ri;
        di = ri + di * beta;
        d[i] = di;
    }
}

template<class real, int blockSize>
__global__ void MergedKernels_cgOp3First_kernel(unsigned int n, float alpha, float beta,
                                           float* r, float* a, float* d, const float* q, const float* b)
{
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = gridDim.x*(blockSize);
    for (unsigned int i = blockIdx.x*(blockSize) + tid; i < n; i += gridSize)
    {
        real qi = q[i];
        real di = b[i];
        real ai = di * alpha;
        a[i] = ai;
        real ri = di - qi * alpha;
        r[i] = ri;
        di = ri + di * beta;
        d[i] = di;
    }
}

//////////////////////
// CPU-side methods //
//////////////////////

enum { DELTA_BSIZE = 64 };
enum { DELTA_GRIDSIZE = 256 };

int CudaMergedKernels3f_cgDeltaTmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+DELTA_BSIZE-1)/DELTA_BSIZE;
    if (nblocs > DELTA_GRIDSIZE) nblocs = DELTA_GRIDSIZE;
    return nblocs;
}

void CudaMergedKernels3f_cgDelta(bool first, unsigned int size, float* delta, float alpha,
                                 void* r, void* a, const void* q, const void* d,
                                 void* tmp, float* cputmp)
{
    int nblocs = CudaMergedKernels3f_cgDeltaTmpSize(size);
    size *= 3;
    if (size==0)
    {
        *delta = 0.0f;
        return;
    }


    dim3 threads(DELTA_BSIZE,1);
    dim3 grid(nblocs,1);
    if (first)
        MergedKernels_cgDelta_kernel<float, true , DELTA_BSIZE > <<< grid, threads >>>(size, alpha, (float*)r, (float*)a, (const float*)q, (const float*)d, (float*)tmp);
    else
        MergedKernels_cgDelta_kernel<float, false, DELTA_BSIZE> <<< grid, threads >>>(size, alpha, (float*)r, (float*)a, (const float*)q, (const float*)d, (float*)tmp);
    
    cudaMemcpy(cputmp,tmp,nblocs*sizeof(float),cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i=0;i<nblocs;i++)
        sum+=cputmp[i];
    *delta = sum;
}

enum { DOT3_BSIZE = 64 };
enum { DOT3_GRIDSIZE = 256 };

int CudaMergedKernels3f_cgDot3TmpSize(unsigned int size)
{
    size *= 3;
    int nblocs = (size+DOT3_BSIZE-1)/DOT3_BSIZE;
    if (nblocs > DOT3_GRIDSIZE) nblocs = DOT3_GRIDSIZE;
    return nblocs*3;
}

void CudaMergedKernels3f_cgDot3(unsigned int size, float* dot3,
                                const void* r, const void* q, const void* d,
                                void* tmp, float* cputmp)
{
    size *= 3;
    int nblocs = (size+DOT3_BSIZE-1)/DOT3_BSIZE;
    if (nblocs > DOT3_GRIDSIZE) nblocs = DOT3_GRIDSIZE;
    if (size==0)
    {
        dot3[0] = 0.0f;
        dot3[1] = 0.0f;
        dot3[2] = 0.0f;
        return;
    }

    dim3 threads(DOT3_BSIZE,1);
    dim3 grid(nblocs,1);
    MergedKernels_cgDot3_kernel<float, false, DOT3_BSIZE> <<< grid, threads >>>(size, (const float*)r, (const float*)q, (const float*)d, (float*)tmp);
    
    cudaMemcpy(cputmp,tmp,nblocs*3*sizeof(float),cudaMemcpyDeviceToHost);
    float sum[3] = {0.0f,0.0f,0.0f};
    for (int i=0;i<nblocs;i++)
    {
        sum[0]+=*(cputmp++);
        sum[1]+=*(cputmp++);
        sum[2]+=*(cputmp++);
    }
    dot3[0] = sum[0];
    dot3[1] = sum[1];
    dot3[2] = sum[2];
}
void CudaMergedKernels3f_cgDot3First(unsigned int size, float* dot3,
                                     const void* b, const void* q,
                                     void* tmp, float* cputmp)
{
    size *= 3;
    int nblocs = (size+DOT3_BSIZE-1)/DOT3_BSIZE;
    if (nblocs > DOT3_GRIDSIZE) nblocs = DOT3_GRIDSIZE;
    if (size==0)
    {
        dot3[0] = 0.0f;
        dot3[1] = 0.0f;
        dot3[2] = 0.0f;
        return;
    }

    dim3 threads(DOT3_BSIZE,1);
    dim3 grid(nblocs,1);
    MergedKernels_cgDot3_kernel<float, true, DOT3_BSIZE> <<< grid, threads >>>(size, (const float*)b, (const float*)q, (const float*)b, (float*)tmp);
    
    cudaMemcpy(cputmp,tmp,nblocs*3*sizeof(float),cudaMemcpyDeviceToHost);
    float sum[3] = {0.0f,0.0f,0.0f};
    for (int i=0;i<nblocs;i++)
    {
        sum[0]+=*(cputmp++);
        sum[1]+=*(cputmp++);
        sum[2]+=*(cputmp++);
    }
    dot3[0] = sum[0];
    dot3[1] = sum[1];
    dot3[2] = sum[2];
}

enum { OP3_BSIZE = 64 };
enum { OP3_GRIDSIZE = 256 };

void CudaMergedKernels3f_cgOp3(unsigned int size, float alpha, float beta,
                               void* r, void* a, void* d, const void* q)
{
    size *= 3;
    int nblocs = (size+OP3_BSIZE-1)/OP3_BSIZE;
    if (nblocs > OP3_GRIDSIZE) nblocs = OP3_GRIDSIZE;

    dim3 threads(OP3_BSIZE,1);
    dim3 grid(nblocs,1);
    MergedKernels_cgOp3_kernel<float, OP3_BSIZE> <<< grid, threads >>>(size, alpha, beta, (float*)r, (float*)a, (float*)d, (const float*)q);
}

void CudaMergedKernels3f_cgOp3First(unsigned int size, float alpha, float beta,
                                    void* r, void* a, void* d, const void* q, const void* b)
{
    size *= 3;
    int nblocs = (size+OP3_BSIZE-1)/OP3_BSIZE;
    if (nblocs > OP3_GRIDSIZE) nblocs = OP3_GRIDSIZE;

    dim3 threads(OP3_BSIZE,1);
    dim3 grid(nblocs,1);
    MergedKernels_cgOp3First_kernel<float, OP3_BSIZE> <<< grid, threads >>>(size, alpha, beta, (float*)r, (float*)a, (float*)d, (const float*)q, (const float*)b);
}

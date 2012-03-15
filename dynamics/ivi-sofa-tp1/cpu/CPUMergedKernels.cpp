#include "../kernels.h"

#if defined(MERGE_REDUCTION_KERNELS)
#ifdef PARALLEL_REDUCTION
int CPUMergedKernels3f_cgDot3TmpSize( unsigned int size )
{
    return 0;
}
#endif
// d.q, r.q, q.q
void CPUMergedKernels3f_cgDot3( unsigned int size, float* dot3
    , const TDeriv* r, const TDeriv* q, const TDeriv* d
#ifdef PARALLEL_REDUCTION
    , TReal* tmp, float* cputmp
#endif
)
{
    TReal dot_dq = 0.0f;
    TReal dot_qq = 0.0f;
    TReal dot_rq = 0.0f;
    for (unsigned int i=0; i<size; ++i)
    {
        TDeriv di = d[i];
        TDeriv qi = q[i];
        TDeriv ri = r[i];
        dot_dq += di * qi;
        dot_rq += ri * qi;
        dot_qq += qi * qi;
    }
    dot3[0] = dot_dq;
    dot3[1] = dot_rq;
    dot3[2] = dot_qq;
}
// b.q, b.q, q.q
void CPUMergedKernels3f_cgDot3First( unsigned int size, float* dot3
    , const TDeriv* b, const TDeriv* q
#ifdef PARALLEL_REDUCTION
    , TReal* tmp, float* cputmp
#endif
)
{
    TReal dot_bq = 0.0f;
    TReal dot_qq = 0.0f;
    for (unsigned int i=0; i<size; ++i)
    {
        TDeriv bi = b[i];
        TDeriv qi = q[i];
        dot_bq += bi * qi;
        dot_qq += qi * qi;
    }
    dot3[0] = dot_bq;
    dot3[1] = dot_bq;
    dot3[2] = dot_qq;
}

// a = a + alpha d, r = r - alpha q, d = r + beta d
void CPUMergedKernels3f_cgOp3( unsigned int size, float alpha, float beta
    , TDeriv* r, TDeriv* a, TDeriv* d, const TDeriv* q
)
{
    for (unsigned int i=0; i<size; ++i)
    {
        TDeriv di = d[i];
        TDeriv qi = q[i];
        TDeriv ai = a[i];
        TDeriv ri = r[i];
        ai += di * alpha;
        a[i] = ai;
        ri -= qi * alpha;
        r[i] = ri;
        di = ri + di * beta;
        d[i] = di;
    }
}

// a = alpha b, r = b - alpha q, d = r + beta b
void CPUMergedKernels3f_cgOp3First( unsigned int size, float alpha, float beta
    , TDeriv* r, TDeriv* a, TDeriv* d, const TDeriv* q
    , const TDeriv* b
)
{
    for (unsigned int i=0; i<size; ++i)
    {
        TDeriv bi = b[i];
        TDeriv qi = q[i];
        TDeriv ai = bi * alpha;
        a[i] = ai;
        TDeriv ri = bi - qi * alpha;
        r[i] = ri;
        TDeriv di = ri + bi * beta;
        d[i] = di;
    }
}

#elif defined(MERGE_CG_KERNELS)
#ifdef PARALLEL_REDUCTION
int CPUMergedKernels3f_cgDeltaTmpSize( unsigned int size )
{
    return 0;
}
#endif

void CPUMergedKernels3f_cgDelta( bool first, unsigned int size, float* delta, float alpha
                                           , TDeriv* r, TDeriv* a, const TDeriv* q, const TDeriv* d
#ifdef PARALLEL_REDUCTION
                                           , TReal* tmp, float* cputmp
#endif
)
{
    TReal sum = 0.0f;
    for (unsigned int i=0; i<size; ++i)
    {
        TDeriv di = d[i];
        TDeriv qi = q[i];
        TDeriv ai = (first) ? TDeriv() : a[i];
        TDeriv ri = (first) ? di : r[i];
        ai += di * alpha;
        a[i] = ai;
        ri -= qi * alpha;
        r[i] = ri;
        sum += ri * ri;
    }
    *delta = sum;
}
#endif

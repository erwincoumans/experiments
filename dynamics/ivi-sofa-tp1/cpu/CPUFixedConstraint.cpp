#include "../kernels.h"

void CPUFixedConstraint3f_projectResponseIndexed( unsigned int size, const int* indices, TDeriv* dx )
{
    for (unsigned int i=0;i<size;++i)
        dx[indices[i]].clear();
}

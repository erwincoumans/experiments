#include "../kernels.h"

void CPUTetraMapper3f_apply( unsigned int size, const TTetra* map_i, const TCoord4* map_f, TDeriv* out, const TDeriv* in )
{
    for (unsigned int i=0;i<size;++i)
    {
        out[i] = 
            in[map_i[i][0]] * map_f[i][0] +
            in[map_i[i][1]] * map_f[i][1] +
            in[map_i[i][2]] * map_f[i][2] +
            in[map_i[i][3]] * map_f[i][3];
    }
}

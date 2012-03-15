#include "../kernels.h"

void CPUVisualModel3f_calcTNormals( unsigned int nbElem, unsigned int nbVertex, const TTriangle* elems, TDeriv* fnormals, const TDeriv* x );
#ifdef PARALLEL_GATHER
void CPUVisualModel3f_calcVNormals( unsigned int nbElem, unsigned int nbVertex, unsigned int nbElemPerVertex, const int* velems, TDeriv* vnormals, const TDeriv* fnormals, const TDeriv* x );
#endif
void CPUVisualModel3f_calcNormals( unsigned int nbElem, unsigned int nbVertex
                                             , const TTriangle* elems, const TDeriv* x
                                             , TDeriv* fnormals, TDeriv* vnormals
#ifdef PARALLEL_GATHER
                                             , unsigned int nbElemPerVertex, const int* velems
#endif
);


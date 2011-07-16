#ifndef OPENTISSUE_DYNAMICS_FEM_FEM_EXPORT_K_H
#define OPENTISSUE_DYNAMICS_FEM_FEM_EXPORT_K_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

namespace OpenTissue
{
  namespace fem
  {
    /**
    * Export Assembled Stiffness Matrix to a Sparse Matrix format (UBLAS style).
    *
    * @param mesh   The mesh containing the assembled K matrix (animate_warp
    *               shold have been invoked on priory to ensure this).
    * @param bigK   Upon return holds the output values.
    */
    template<typename fem_mesh,typename matrix_type>
    inline void export_K(fem_mesh & mesh,matrix_type & bigK)
    {
      typedef typename fem_mesh::real_type                     real_type;
      typedef typename fem_mesh::vector3_type                  vector3_type;
      typedef typename fem_mesh::matrix3x3_type                matrix3x3_type;
      typedef typename fem_mesh::node_iterator                 node_iterator;
      typedef typename fem_mesh::node_type::matrix_iterator    matrix_iterator;

      unsigned int N = mesh.size_nodes();
      bigK.resize(N*3,N*3, false );
      bigK.clear();

      unsigned int row = 0;
      for(unsigned int i=0;i<N;++i)
      {
        node_iterator n_i = mesh.node(i);
        for(unsigned int r=0;r<3;++r,++row)
        {
          if(n_i->m_fixed)
          {
            bigK.push_back(row ,row, 1.0);
          }
          else
          {
            matrix_iterator Kbegin = n_i->Kbegin();
            matrix_iterator Kend   = n_i->Kend();
            for (matrix_iterator K = Kbegin; K != Kend;++K)
            {
              unsigned int     j    = K->first;
              node_iterator    n_j  = mesh.node(j);

              if(n_j->m_fixed)
                continue;

              matrix3x3_type & K_ij = K->second;

              unsigned int column = j*3;
              for(unsigned int c=0;c<3;++c)
                bigK.push_back(row,column + c, K_ij(r,c));
            }
          }
        }

      }
    }

  } // namespace fem
} // namespace OpenTissue

//OPENTISSUE_DYNAMICS_FEM_FEM_EXPORT_K_H
#endif

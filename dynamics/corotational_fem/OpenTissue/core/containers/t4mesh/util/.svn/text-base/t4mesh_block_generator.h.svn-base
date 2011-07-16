#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_BLOCK_GENERATOR_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_BLOCK_GENERATOR_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/containers/grid/grid.h>
#include <OpenTissue/core/containers/grid/util/grid_idx2coord.h>

namespace OpenTissue
{
  namespace t4mesh
  {


    /**
    * t4mesh Block Generator.
    *
    * @param I               The number of blocks along x axis.
    * @param J               The number of blocks along y axis.
    * @param K               The number of blocks along z axis.
    * @param block_width     The edgelength of the blocks along x-axis.
    * @param block_height    The edgelength of the blocks along x-axis.
    * @param block_depth     The edgelength of the blocks along x-axis.
    * @param mesh            A generic t4mesh, which upon return holds the generated mesh.
    */
    template < typename real_type, typename t4mesh_type >
    void generate_blocks(
      unsigned int const & I
      , unsigned int const & J
      , unsigned int const & K
      , real_type const & block_width
      , real_type const & block_height
      , real_type const & block_depth
      , t4mesh_type & mesh
      )
    {
      typedef typename t4mesh_type::node_iterator          node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator   tetrahedron_iterator;

      mesh.clear();

      unsigned int numVertices = (I + 1) * (J + 1) * (K + 1);
      for (unsigned int i=0; i<numVertices; ++i)
        mesh.insert();

      node_iterator node = mesh.node_begin();
      for (unsigned int x = 0; x <= I; ++x)
      {
        for (unsigned int y = 0; y <= J; ++y)
        {
          for (unsigned int z = 0; z <= K; ++z)
          {
            node->m_coord(0) = block_width*x;
            node->m_coord(2) = block_depth*y;
            node->m_coord(1) = block_height*z;
            ++node;
          }
        }
      }
      for (unsigned int i = 0; i < I; ++i)
      {
        for (unsigned int j = 0; j < J; ++j)
        {
          for (unsigned int k = 0; k < K; ++k)
          {
            // For each block, the 8 corners are numerated as:
            //     4*-----*7
            //     /|    /|
            //    / |   / |
            //  5*-----*6 |
            //   | 0*--|--*3
            //   | /   | /
            //   |/    |/
            //  1*-----*2
            int p0 = (i * (J + 1) + j) * (K + 1) + k;
            int p1 = p0 + 1;
            int p3 = ((i + 1) * (J + 1) + j) * (K + 1) + k;
            int p2 = p3 + 1;
            int p7 = ((i + 1) * (J + 1) + (j + 1)) * (K + 1) + k;
            int p6 = p7 + 1;
            int p4 = (i * (J + 1) + (j + 1)) * (K + 1) + k;
            int p5 = p4 + 1;
            tetrahedron_iterator t;
            // Ensure that neighboring tetras are sharing faces
            if ((i + j + k) % 2 == 1)
            {
              t = mesh.insert(p1,p2,p6,p3);
              t = mesh.insert(p3,p6,p4,p7);
              t = mesh.insert(p1,p4,p6,p5);
              t = mesh.insert(p1,p3,p4,p0);
              t = mesh.insert(p1,p6,p4,p3);
            }
            else
            {
              t = mesh.insert(p2,p0,p5,p1);
              t = mesh.insert(p2,p7,p0,p3);
              t = mesh.insert(p2,p5,p7,p6);
              t = mesh.insert(p0,p7,p5,p4);
              t = mesh.insert(p2,p0,p7,p5);
            }
          }
        }
      }
    };


    /**
    * t4mesh Block Generator for voxelized data.
    *
    * @param voxels          A voxelized map, indicating where to generate tetrahedra.
    * @param mesh            A generic t4mesh, which upon return holds the generated mesh.
    */
    template <typename t4mesh_type, typename grid_type >
    void generate_blocks(
      grid_type & voxels
      , t4mesh_type & mesh
      )
    {
      typedef typename grid_type::math_types                math_types;
      typedef typename t4mesh_type::node_iterator          node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator   tetrahedron_iterator;
      typedef typename math_types::real_type               real_type;
      typedef typename math_types::vector3_type            vector3_type;
      typedef OpenTissue::grid::Grid<int,math_types>                          idx_grid_type;
      typedef typename grid_type::index_iterator            voxel_iterator;

      mesh.clear();

      idx_grid_type grid;
      grid.create(voxels.min_coord(),voxels.max_coord(),voxels.I(),voxels.J(),voxels.K());

      //--- First pass, create t4mesh nodes
      {
        voxel_iterator v = voxels.begin();
        voxel_iterator vend = voxels.end();
        for(;v!=vend;++v)
        {
          if(*v)
          {
            vector3_type coord;
            OpenTissue::grid::idx2coord(v,coord);
            node_iterator n = mesh.insert( coord );
            grid(v.i(),v.j(),v.k()) = n->idx();
          }
          else
            grid(v.i(),v.j(),v.k()) = -1;
        }
      }

      //--- Second pass, create tetrahedra in t4mesh
      unsigned int I            = voxels.I();
      unsigned int J            = voxels.J();
      unsigned int K            = voxels.K();
      for (unsigned int i = 0; i < I-1; ++i)
      {
        for (unsigned int j = 0; j < J-1; ++j)
        {
          for (unsigned int k = 0; k < K-1; ++k)
          {
            // For each block, the 8 corners are numerated as:
            //
            //
            //        4*-----*7
            //        /|    /|
            //       / |   / | K
            //     5*-----*6 |
            //      | 0*--|--*3
            //      | /   | /
            //      |/    |/ I
            //     1*-----*2
            //         J
            //
            //
            //
            int p0 =  (k*J + j)*I  + i;
            int p1 =  (k*J + j)*I  + (i+1);
            int p2 =  (k*J + (j+1))*I  + (i+1);
            int p3 =  (k*J + (j+1))*I  + i;          
            int p4 =  ((k+1)*J + j)*I  + i;
            int p5 =  ((k+1)*J + j)*I  + (i+1);
            int p6 =  ((k+1)*J + (j+1))*I  + (i+1);
            int p7 =  ((k+1)*J + (j+1))*I  + i;

            tetrahedron_iterator t;

            //--- Ensure that neighboring tetrahedra are sharing faces
            if ((i + j + k) % 2 == 1)
            {
              if ( voxels(p1) && voxels(p2) && voxels(p3) && voxels(p6) )
                t = mesh.insert(grid(p1),grid(p2),grid(p3),grid(p6));

              if ( voxels(p3) && voxels(p6) && voxels(p7) && voxels(p4) )
                t = mesh.insert(grid(p3),grid(p6),grid(p7),grid(p4));

              if ( voxels(p1) && voxels(p4) && voxels(p5) && voxels(p6) )
                t = mesh.insert(grid(p1),grid(p4),grid(p5),grid(p6));

              if ( voxels(p1) && voxels(p3) && voxels(p0) && voxels(p4) )
                t = mesh.insert(grid(p1),grid(p3),grid(p0),grid(p4));

              if ( voxels(p1) && voxels(p6) && voxels(p3) && voxels(p4) )
                t = mesh.insert(grid(p1),grid(p6),grid(p3),grid(p4));
            }
            else
            {
              if ( voxels(p2) && voxels(p0) && voxels(p1) && voxels(p5) )
                t = mesh.insert(grid(p2),grid(p0),grid(p1),grid(p5));

              if ( voxels(p2) && voxels(p7) && voxels(p3) && voxels(p0) )
                t = mesh.insert(grid(p2),grid(p7),grid(p3),grid(p0));

              if ( voxels(p2) && voxels(p5) && voxels(p6) && voxels(p7) )
                t = mesh.insert(grid(p2),grid(p5),grid(p6),grid(p7));

              if ( voxels(p0) && voxels(p7) && voxels(p4) && voxels(p5) )
                t = mesh.insert(grid(p0),grid(p7),grid(p4),grid(p5));

              if ( voxels(p2) && voxels(p0) && voxels(p5) && voxels(p7) )
                t = mesh.insert(grid(p2),grid(p0),grid(p5),grid(p7));
            }
          }
        }
      }

      //--- Third pass erase isolated tetrahedra nodes!!!
      node_iterator n = mesh.node_begin();
      node_iterator nend = mesh.node_end();
      for(;n!=nend;++n)
      {
        if(n->isolated())
        {
          std::cout << "t4mesh::generate_blocks(): Isolated node encountered, could not erase it from t4mesh" << std::endl;
          //mesh.erase(n); //--- sorry not implemented yet!!!
        }
      }
      std::cout << "t4mesh::generate_blocks(): |N| = " << mesh.size_nodes() << " |T| = " << mesh.size_tetrahedra() << std::endl;
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_BLOCK_GENERATOR_H
#endif

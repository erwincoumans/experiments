#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_DELAUNAY_TETRAHEDRALIZATION_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_DELAUNAY_TETRAHEDRALIZATION_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/utility_qhull.h>

#include <iostream>
#include <cassert>
#include <map>

namespace OpenTissue
{
  namespace t4mesh
  {


    /**
    * Delaynay Tesselator.
    * This utility method provides a wrapper interface for QHull, which
    * performs a delaynay triangulation a given set of points.
    *
    * @param points     A container of Vector3<double> points, these will be the nodes in the
    *                   resulting tetrahedral mesh.
    * @param mesh       A tetrahedral mesh, upon return this will contain the resulting
    *                   triangulation. It is expected that the mesh data structure conforms
    *                   with the t4mesh.
    */
    template <typename Points, typename t4mesh_type>
    void delaunay_tetrahedralization( Points & points, t4mesh_type & output )
    {
      output.clear();
      int N = static_cast<int>( points.size() );
      if ( N == 0 )
        return ;

      assert( N > 3 );
      int dim = 3;
      coordT *coords = new coordT[ dim * N ];
      std::map<unsigned int, typename t4mesh_type::node_iterator > nodeMap;
      int i = 0;
      int j = 0;
      for ( typename Points::iterator point = points.begin();point != points.end();++point )
      {
        typename t4mesh_type::node_iterator node = output.insert();
        node->m_coord = *point;
        coords[ j++ ] = ( *point )(0);
        coords[ j++ ] = ( *point )(1);
        coords[ j++ ] = ( *point )(2);
        nodeMap[ i++ ] = node;
      }
      boolT ismalloc = False;
      char flags[] = "qhull d QJ";
      //FILE *outfile = stdout;
      //FILE *errfile = stderr;
      FILE *outfile = 0;
      FILE *errfile = 0;
      int exitcode;
      exitcode = qh_new_qhull ( dim, N, coords, ismalloc, flags, outfile, errfile );
      if ( !exitcode )
      {
        facetT * facet;
        vertexT *vertex, **vertexp;
        FORALLfacets {
          if ( !facet->upperdelaunay )
          {
            assert( qh_setsize ( facet->vertices ) == 4 );
            typename t4mesh_type::node_iterator tmp[ 4 ];
            int j = 0;
            if ( !facet->toporient )
            {
              // TODO: Compiler warning (VC++): assignment within conditional expression
              FOREACHvertexreverse12_( facet->vertices )
              {
                int i = qh_pointid ( vertex->point );
                tmp[ j++ ] = nodeMap[ i ];
              }
            }
            else
            {
              FOREACHvertex_( facet->vertices )
              {
                int i = qh_pointid ( vertex->point );
                tmp[ j++ ] = nodeMap[ i ];
              }
            }
            output.insert( tmp[ 0 ], tmp[ 1 ], tmp[ 2 ], tmp[ 3 ] );
          }
        }
      }
      qh_freeqhull( !qh_ALL );
      int curlong, totlong;
      qh_memfreeshort ( &curlong, &totlong );
      if ( curlong || totlong )
        fprintf ( errfile, "qhull internal warning (main): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong );
      delete [] coords;

      std::cout << "Delaunay resulted in " << output.size_tetrahedra() << " tetrahedra with " << output.size_nodes() << " nodes" << std::endl;
    };


    /**
    * Delaynay Surface Tetrahedralization.
    *
    * @param surface   The input surface. The vertex coordinates are extracted and used for making a Delaunay tetrahedralization.
    * @param output    Upon return this argument holds the resulting Delaynay tetrahedralization.
    */
    template <typename mesh_type, typename t4mesh_type>
    void delaunay_from_surface_tetrahedralization( mesh_type const & input, t4mesh_type & output )
    {
      typedef typename mesh_type::math_types    math_types;
      typedef typename math_types::vector3_type vector3_type;

      std::vector< vector3_type > points;

      output.clear();

      if( input.size_vertices() <= 0 )
        return;

      points.resize( input.size_vertices() );
      for( typename mesh_type::const_vertex_iterator v = input.vertex_begin();v!=input.vertex_end();++v)
      {
        points[v->get_handle().get_idx()] = v->m_coord;
      }
      delaunay_tetrahedralization( points, output);
    }

  } // namespace t4mesh
} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_DELAUNAY_TETRAHEDRALIZATION_H
#endif

#ifndef OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_COMPUTE_QUALITY_MESH_H
#define OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_COMPUTE_QUALITY_MESH_H
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
  namespace t4mesh
  {

    /**
     * Compute Mesh Quality Vector.
     *
     * @param mesh      A tetrahedra mesh containing the topology of the tetrahedrons.
     * @param points    A containing holding the coordinates of the tetrahedra nodes.
     * @param Q         Upon return each entry holds the quality measure of a thetrahedron.
     * @param F         Quality measure functor. 
     */
    template<typename point_container,typename t4mesh_type, typename vector_type, typename quality_functor>
    inline void compute_mesh_quality(t4mesh_type const & mesh, point_container & points, vector_type & Q, quality_functor const & F)
    {
      typedef typename t4mesh_type::node_iterator node_iterator;
      typedef typename t4mesh_type::node_iterator const_node_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator tetrahedron_iterator;
      typedef typename t4mesh_type::tetrahedron_iterator const_tetrahedron_iterator;

      Q.clear();
      Q.resize( mesh.size_tetrahedra() );

      size_t i=0;
      for(const_tetrahedron_iterator tetrahedron=mesh.tetrahedron_begin();tetrahedron!=mesh.tetrahedron_end();++tetrahedron)
      {          
        Q[i++] = F(
          points[tetrahedron->i()->idx()]
        , points[tetrahedron->j()->idx()]
        , points[tetrahedron->k()->idx()]
        , points[tetrahedron->m()->idx()]
        );
      }
    }

    /**
     * Compute Mesh Quality Vector.
     *
     * @param mesh      A tetrahedra mesh containing the topology of the tetrahedrons.
     * @param Q         Upon return each entry holds the quality measure of a thetrahedron.
     * @param F         Quality measure functor. 
     */
    template<typename t4mesh_type, typename vector_type, typename quality_functor>
    inline void compute_mesh_quality(t4mesh_type const & mesh, vector_type & Q, quality_functor const & F)
    {
      default_point_container<t4mesh_type> points(const_cast<t4mesh_type*>(&mesh));
      return compute_mesh_quality(mesh,points,Q,F);
    }

  } // namespace t4mesh

} // namespace OpenTissue

//OPENTISSUE_CORE_CONTAINERS_T4MESH_UTIL_T4MESH_COMPUTE_QUALITY_MESH_H
#endif

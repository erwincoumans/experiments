#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_T4MESH_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_T4MESH_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_basic_types.h>
#include <OpenTissue/core/geometry/geometry_tetrahedron.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw t4mesh.
    *
    * Assumes that nodes have default memeber m_coord
    *
    * @param mesh       The mesh that should be drawn.
    * @param scale      A scale of each individual tetrahedron that is drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template <typename t4mesh>
    inline void DrawT4Mesh(t4mesh const& mesh, double scale = 0.95, bool wireframe = false)
    {
      typedef typename t4mesh::const_node_iterator              const_node_iterator;
      typedef typename t4mesh::const_tetrahedron_iterator       const_tetrahedron_iterator;

      geometry::Tetrahedron<math::default_math_types> T; // From OpenTissue/core/geometry/geometry_tetrahederon.h

      for (const_tetrahedron_iterator tetrahedron = mesh.tetrahedron_begin();tetrahedron != mesh.tetrahedron_end(); ++tetrahedron)
      {
        const_node_iterator i = tetrahedron->i();
        const_node_iterator j = tetrahedron->j();
        const_node_iterator k = tetrahedron->k();
        const_node_iterator m = tetrahedron->m();
        T.p0() = i->m_coord ;
        T.p1() = j->m_coord ;
        T.p2() = k->m_coord ;
        T.p3() = m->m_coord ;
        T.scale( scale );
        DrawTetrahedron( T, wireframe );
      }
    };


    /**
    *
    * @param points     A point container with coordinates of the nodes.
    * @param mesh       The mesh that should be drawn.
    * @param scale      A scale of each individual tetrahedron that is drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template <typename point_container, typename t4mesh >
    inline void DrawPointsT4Mesh( point_container const& points, t4mesh const& mesh, double const& scale = 0.95, bool wireframe = false)
    {
      geometry::Tetrahedron<math::default_math_types> T; // From OpenTissue/core/geometry/geometry_tetrahederon.h

      for (int t=0;t<mesh.m_tetrahedra.size();t++)
      {
        T.p0() = points[mesh.m_tetrahedra[t].m_nodes[0]];
        T.p1() = points[mesh.m_tetrahedra[t].m_nodes[1]];
        T.p2() = points[mesh.m_tetrahedra[t].m_nodes[2]];
        T.p3() = points[mesh.m_tetrahedra[t].m_nodes[3]];
        T.scale( scale );
        DrawTetrahedron( T, wireframe );
      }
    };


    /**
    * Draw t4mesh cut through.
    *
    * Assumes that nodes have default memeber m_coord
    *
    * @param mesh       The mesh that should be drawn.
    * @param plane      The plane that defines the cut.
    * @param scale      A scale of each individual tetrahedron that is drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template <typename t4mesh_type, typename plane_type >
    inline void DrawT4MeshCutThrough(t4mesh_type const& mesh, plane_type const& plane, double scale = 1.0, bool wireframe = false)
    {
      typedef typename t4mesh_type::const_node_iterator              const_node_iterator;
      typedef typename t4mesh_type::const_tetrahedron_iterator       const_tetrahedron_iterator;

      geometry::Tetrahedron<math::default_math_types> T;

      const_tetrahedron_iterator tend = mesh.tetrahedron_end();
      for (const_tetrahedron_iterator tetrahedron = mesh.tetrahedron_begin(); tetrahedron != tend; ++tetrahedron)
      {
        const_node_iterator i = tetrahedron->i();
        const_node_iterator j = tetrahedron->j();
        const_node_iterator k = tetrahedron->k();
        const_node_iterator m = tetrahedron->m();
        if (plane.signed_distance(i->m_coord) <= 0 ||
          plane.signed_distance(j->m_coord) <= 0 ||
          plane.signed_distance(k->m_coord) <= 0 ||
          plane.signed_distance(m->m_coord) <= 0) continue;
        T.p0() = i->m_coord ;
        T.p1() = j->m_coord ;
        T.p2() = k->m_coord ;
        T.p3() = m->m_coord ;
        T.scale( scale );
        DrawTetrahedron( T, wireframe );
      }
    }


    /**
    * Draw t4mesh cut through.
    *
    * Assumes that nodes have default memeber m_coord
    *
    * @param points     A point container with coordinates of the nodes.
    * @param mesh       The mesh that should be drawn.
    * @param plane      The plane that defines the cut.
    * @param scale      A scale of each individual tetrahedron that is drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template <typename point_container,typename t4mesh_type, typename plane_type >
    inline void DrawPointsT4MeshCutThrough( point_container const& points, t4mesh_type const& mesh, plane_type const& plane, double scale = 1.0, bool wireframe = false)
    {
      typedef typename t4mesh_type::const_node_iterator              const_node_iterator;
      typedef typename t4mesh_type::const_tetrahedron_iterator       const_tetrahedron_iterator;

      geometry::Tetrahedron<math::default_math_types> T;

      const_tetrahedron_iterator tend = mesh.tetrahedron_end();
      for (const_tetrahedron_iterator tetrahedron = mesh.tetrahedron_begin(); tetrahedron != tend; ++tetrahedron)
      {
        const_node_iterator i = tetrahedron->i();
        const_node_iterator j = tetrahedron->j();
        const_node_iterator k = tetrahedron->k();
        const_node_iterator m = tetrahedron->m();
        if (plane.signed_distance( points[i->idx()] ) <= 0 ||
          plane.signed_distance( points[j->idx()] ) <= 0 ||
          plane.signed_distance( points[k->idx()] ) <= 0 ||
          plane.signed_distance( points[m->idx()] ) <= 0) continue;
        T.p0() = points[i->idx()] ;
        T.p1() = points[j->idx()] ;
        T.p2() = points[k->idx()] ;
        T.p3() = points[m->idx()] ;
        T.scale( scale );
        DrawTetrahedron( T, wireframe );
      }
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_T4MESH_H
#endif

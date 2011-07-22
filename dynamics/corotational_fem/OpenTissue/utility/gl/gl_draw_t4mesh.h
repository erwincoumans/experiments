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


  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_T4MESH_H
#endif

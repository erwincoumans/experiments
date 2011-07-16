#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_HYBRID_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_HYBRID_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/utility/gl/gl_draw_aabb.h>
#include <OpenTissue/utility/gl/gl_draw_obb.h>
#include <OpenTissue/utility/gl/gl_draw_sphere.h>
#include <OpenTissue/utility/gl/gl_draw_prism.h>
#include <OpenTissue/utility/gl/gl_draw_cylinder.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Hybrid
    * This method draws the hybrid volume in the world coordinate system.
    *
    * @param hybdrid     A reference to the Hybrid that should be drawn.
    * @param wireframe   Draw in wireframe or normal.
    */
    template<typename hybrid_type>
    inline void DrawHybrid(hybrid_type const & hybrid, bool wireframe = false) 
    {
      switch(hybrid.selected_type())
      {
      case hybrid_type::selection_aabb:
        DrawAABB( hybrid.m_aabb, wireframe);
        break;
      case hybrid_type::selection_obb:
        DrawOBB( hybrid.m_obb, wireframe );
        break;
      case hybrid_type::selection_sphere:
        DrawSphere( hybrid.m_sphere, wireframe );
        break;
      case hybrid_type::selection_cylinder:
        DrawCylinder( hybrid.m_cylinder, wireframe );
        break;
      case hybrid_type::selection_prism:
        DrawPrism(hybrid.m_prism, wireframe );
        break;
      case hybrid_type::selection_tetrahedron:
        assert(!"DrawHybrid(): case not handled");
        break;
      case hybrid_type::selection_undefined:
        break;
      }
    };

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_HYBRID_H
#endif

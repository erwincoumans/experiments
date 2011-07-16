#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_BOX_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_BOX_H
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

namespace OpenTissue
{
  namespace gl
  {

    /**
     * Draw Plane Box.
     *
     * @param box        A reference to a plane box object.
     * @param wireframe  A boolean indicating whether the plane box should be visualized as wireframe or solid.
     */
    template< typename plane_box_type>
    inline void DrawPlaneBox( plane_box_type const & box, bool wireframe = true )
    {
      DrawAABB(box.box(),true);

      if(wireframe)
        glBegin(GL_LINE_LOOP);
      else
        glBegin(GL_POLYGON);
      glNormal3f( box.n()[0],box.n()[1],box.n()[2]);
      glVertex3f( box.p0()[0],box.p0()[1],box.p0()[2]);
      glNormal3f( box.n()[0],box.n()[1],box.n()[2]);
      glVertex3f( box.p1()[0],box.p1()[1],box.p1()[2]);
      glNormal3f( box.n()[0],box.n()[1],box.n()[2]);
      glVertex3f( box.p2()[0],box.p2()[1],box.p2()[2]);
      glNormal3f( box.n()[0],box.n()[1],box.n()[2]);
      glVertex3f( box.p3()[0],box.p3()[1],box.p3()[2]);
      glEnd();
    }
  }

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_BOX_H
#endif

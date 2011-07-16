#ifndef OPENTISSUE_OPENTISSUE_GL_GL_DRAW_TRIANGLE_H
#define OPENTISSUE_OPENTISSUE_GL_GL_DRAW_TRIANGLE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

namespace OpenTissue
{

  namespace gl
  {
    
    /**
    * Drawing Routine
    *
    * @param triangle   A reference to the triangle that should be drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename triangle_type>
    inline void DrawTriangle(triangle_type const & triangle, bool wireframe = false)
    {
      typedef typename triangle_type::vector3_type vector3_type;

      vector3_type const & p0 = triangle.p0();
      vector3_type const & p1 = triangle.p1();
      vector3_type const & p2 = triangle.p2();

      vector3_type n,v,u;
      v = p1 - p0;
      u = p2 - p1;
      n = unit( v % u);
      glBegin(wireframe?GL_LINE_LOOP:GL_POLYGON);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glEnd();
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_OPENTISSUE_GL_GL_DRAW_TRIANGLE_H
#endif

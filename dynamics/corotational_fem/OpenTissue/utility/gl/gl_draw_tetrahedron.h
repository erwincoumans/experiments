#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_TETRAHEDRON_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_TETRAHEDRON_H
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
    * @param tetrahedron  A reference to the tetrahedron that should be drawn.
    * @param wireframe    Draw in wireframe or normal.
    */
    template<typename tetrahedron_type>
    inline void DrawTetrahedron(tetrahedron_type const & tetrahedron, bool wireframe = false)
    {
      typedef typename tetrahedron_type::vector3_type   vector3_type;

      vector3_type const & p0 = tetrahedron.p0();
      vector3_type const & p1 = tetrahedron.p1();
      vector3_type const & p2 = tetrahedron.p2();
      vector3_type const & p3 = tetrahedron.p3();

      GLenum const mode = wireframe ? GL_LINE_LOOP : GL_POLYGON ;

      vector3_type n, v, u;

      v = p0 - p1;
      u = p2 - p1;
      n = unit(v % u);
      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glEnd();

      v = p1 - p0;
      u = p3 - p0;
      n = unit(v % u);
      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p3[0],(GLfloat)p3[1],(GLfloat)p3[2]);
      glEnd();


      v = p2 - p1;
      u = p3 - p1;
      n = unit(v % u);
      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p3[0],(GLfloat)p3[1],(GLfloat)p3[2]);
      glEnd();

      v = p0 - p2;
      u = p3 - p2;
      n = unit(v % u);
      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p3[0],(GLfloat)p3[1],(GLfloat)p3[2]);
      glEnd();
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_TETRAHEDRON_H
#endif

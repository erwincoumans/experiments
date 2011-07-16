#ifndef OPENTISSUE_OPENTISSUE_UTILITY_GL_GL_DRAW_PRISM_H
#define OPENTISSUE_OPENTISSUE_UTILITY_GL_GL_DRAW_PRISM_H
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
    * Prism Drawing Routine
    *
    * @parma prism
    * @param mode        GL_LINE_LOOP or GL_POLYGON mode
    */
    template<typename prism_type>
    inline void DrawPrism(prism_type const & prism, unsigned int mode)
    {
      typedef typename prism_type::real_type       real_type;
      typedef typename prism_type::vector3_type    vector3_type;

      vector3_type const & p0     = prism.p0();
      vector3_type const & p1     = prism.p1();
      vector3_type const & p2     = prism.p2();
      real_type    const & height = prism.height();


      vector3_type u = (p1-p0);
      vector3_type v = (p2-p0);
      vector3_type n = unit ( u % v);

      vector3_type p0h = p0 + height*n;
      vector3_type p1h = p1 + height*n;
      vector3_type p2h = p2 + height*n;

      glBegin(mode);
      glNormal3f((GLfloat)-n[0],(GLfloat)-n[1],(GLfloat)-n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)-n[0],(GLfloat)-n[1],(GLfloat)-n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glNormal3f((GLfloat)-n[0],(GLfloat)-n[1],(GLfloat)-n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glEnd();
      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0h[0],(GLfloat)p0h[1],(GLfloat)p0h[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1h[0],(GLfloat)p1h[1],(GLfloat)p1h[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2h[0],(GLfloat)p2h[1],(GLfloat)p2h[2]);
      glEnd();


      u = (p1-p0);
      v = (p1h-p1);
      n = unit ( u % v);

      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1h[0],(GLfloat)p1h[1],(GLfloat)p1h[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0h[0],(GLfloat)p0h[1],(GLfloat)p0h[2]);
      glEnd();

      u = (p2-p1);
      v = (p2h-p2);
      n = unit ( u % v);

      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1[0],(GLfloat)p1[1],(GLfloat)p1[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2h[0],(GLfloat)p2h[1],(GLfloat)p2h[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p1h[0],(GLfloat)p1h[1],(GLfloat)p1h[2]);
      glEnd();

      u = (p0-p2);
      v = (p0h-p0);
      n = unit ( u % v);

      glBegin(mode);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2[0],(GLfloat)p2[1],(GLfloat)p2[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0[0],(GLfloat)p0[1],(GLfloat)p0[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p0h[0],(GLfloat)p0h[1],(GLfloat)p0h[2]);
      glNormal3f((GLfloat)n[0],(GLfloat)n[1],(GLfloat)n[2]);
      glVertex3f((GLfloat)p2h[0],(GLfloat)p2h[1],(GLfloat)p2h[2]);
      glEnd();
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_OPENTISSUE_UTILITY_GL_GL_DRAW_PRISM_H
#endif

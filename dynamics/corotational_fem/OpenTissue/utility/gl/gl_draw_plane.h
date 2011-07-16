#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_H
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
    * Draw Plane.
    *
    *
    * @param plane      A reference to the plane that should be drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename plane_type>
    inline void DrawPlane(plane_type const & plane, bool wireframe = false)
    {
      typedef typename plane_type::vector3_type vector3_type;
      vector3_type p = plane.n()*plane.w();
      vector3_type s,t;
      plane.compute_plane_vectors(s,t);
      if (wireframe)
      {
        glBegin(GL_LINES);
        vector3_type a = p - 50*t;
        for(unsigned int i=0;i<101;++i)
        {
          vector3_type b = a - 50*s;
          vector3_type c = a + 50*s;
          glVertex3f(b(0),b(1),b(2));
          glVertex3f(c(0),c(1),c(2));
          a+=t;
        }
        a = p - 50*s;
        for(unsigned int i=0;i<101;++i)
        {
          vector3_type b = a - 50*t;
          vector3_type c = a + 50*t;
          glVertex3f(b(0),b(1),b(2));
          glVertex3f(c(0),c(1),c(2));
          a+=s;
        }
        glEnd();
        return;
      }

      // 2007-07-24 micky: the plane is drawn using small quads rather then one huge quad due to Gouraud shading.
      glBegin(GL_QUADS);
      glNormal3f(plane.n()(0), plane.n()(1), plane.n()(2));
      vector3_type r = p - 50*s - 50*t;
      for(int j = -49; j < 51; ++j)
      {
        for(int i = -49; i < 51; ++i)
        {
          glVertex3f(r(0), r(1), r(2));
          r += s;
          glVertex3f(r(0), r(1), r(2));
          r += t;
          glVertex3f(r(0), r(1), r(2));
          r -= s;
          glVertex3f(r(0), r(1), r(2));
          r -= t;
          r += s;
        }
        r -= 100*s;
        r += t;
      }
      glEnd();

    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_PLANE_H
#endif

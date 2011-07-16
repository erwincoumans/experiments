#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_SPHERE_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_SPHERE_H
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
    * Draw Sphere.
    *
    * @param sphere     A reference to the sphere that should be drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename sphere_type>
    inline void DrawSphere(sphere_type const & sphere, bool wireframe = false)
    {
      typedef typename sphere_type::real_type     real_type;
      typedef typename sphere_type::vector3_type  vector3_type;

      real_type    const & r = sphere.radius();
      vector3_type const & c = sphere.center();

      glPolygonMode(GL_FRONT_AND_BACK,(wireframe?GL_LINE:GL_FILL));
      GLUquadricObj* qobj = gluNewQuadric();
      glPushMatrix();
      glTranslatef(
        (GLfloat)c[0],
        (GLfloat)c[1],
        (GLfloat)c[2]
      );
      gluSphere(qobj,r,32,32);
      glPopMatrix();
      gluDeleteQuadric(qobj);
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_SPHERE_H
#endif

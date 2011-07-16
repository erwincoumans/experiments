#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_CAPSULE_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_CAPSULE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/math/math_constants.h>
#include <cmath>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Capsule
    *
    * @param capsule    A reference to the capsule that should be drawn
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename capsule_type>
    inline void DrawCapsule(capsule_type const & capsule, bool wireframe = false) 
    {
      using std::acos;
      typedef typename capsule_type::real_type      real_type;
      typedef typename capsule_type::vector3_type   vector3_type;
      typedef typename capsule_type::value_traits   value_traits;

      glPolygonMode(GL_FRONT_AND_BACK,(wireframe?GL_LINE:GL_FILL));

      vector3_type const & p0     = capsule.point0();
      vector3_type const & p1     = capsule.point1();
      real_type    const & radius = capsule.radius();

      vector3_type ls(p1 - p0);
      vector3_type const za(math::detail::axis_z<real_type>());
      real_type const ln = length(ls);
      if (ln > value_traits::zero())
      {
        ls /= ln;
      }
      else
      {
        ls = za;
      }
      vector3_type const rt(cross(za, ls));
      real_type const an = math::to_degrees<real_type>(acos(za*ls));

      GLdouble plane[4];
      plane[0] = 0.;
      plane[1] = 0.;
      plane[2] = -1.;
      plane[3] = 0.;
      GLUquadricObj* qobj = gluNewQuadric();

      glPushMatrix();
      glTranslated(p0[0], p0[1], p0[2]);
      glRotated(an, rt[0], rt[1], rt[2]);
      glClipPlane(GL_CLIP_PLANE0,plane);
      glEnable(GL_CLIP_PLANE0);
      gluSphere(qobj, radius, 16, 16);
      glDisable(GL_CLIP_PLANE0);
      glPopMatrix();

      glPushMatrix();
      glTranslated(p1[0], p1[1], p1[2]);
      glRotated(an, rt[0], rt[1], rt[2]);
      plane[2] *= -1;
      glClipPlane(GL_CLIP_PLANE0,plane);
      glEnable(GL_CLIP_PLANE0);
      gluSphere(qobj, radius, 16, 16);
      glDisable(GL_CLIP_PLANE0);
      glPopMatrix();

      if (ln > 0.) 
      {
        glPushMatrix();
        glTranslated(p0[0], p0[1], p0[2]);
        glRotated(an, rt[0], rt[1], rt[2]);
        gluCylinder(qobj, radius, radius, ln, 16, 1);
        glPopMatrix();
      }

      gluDeleteQuadric(qobj);

      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_CAPSULE_H
#endif

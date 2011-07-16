#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_ARC_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_ARC_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/math/math_functions.h>


namespace OpenTissue
{

  namespace gl
  {
    
    /**
    * Draw Arc (draws a 2D arc in the xy-plane, rotates about the z-axis)
    *
    * @param x0, x1     The end points of the arc.
    * @parem angle      The (start) angle at the (start) point in radians [
    *                   This angle is relative to the base line segment (x1 - x0).
    * @param segments   Amount of line steps between x0 and x1 (the higher the smoother)
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename vector3_type>
    inline void DrawArc(vector3_type const & x0, vector3_type const & x1, typename vector3_type::value_type const & angle, size_t segments, bool wireframe)
    {
      typedef typename vector3_type::value_traits  value_traits;
      typedef typename vector3_type::value_type    real_type;
      using std::fabs;

      // draw straight line if angle is close to -or zero, or if segments won't allow an arc
      bool const invalid = fabs(angle) < value_traits::one()/4096 || segments < 2;
      if(!wireframe || invalid)
      {
        glBegin(GL_LINES);
        glVertex3d(x0(0), x0(1), 0.0);
        glVertex3d(x1(0), x1(1), 0.0);
        glEnd();
        if(invalid)
        {
          return;
        }
      }

      using std::sin;
      using std::sqrt;
      using std::fabs;

      real_type const l = math::length(x1-x0);
      real_type const r = l/(value_traits::two()*sin(fabs(angle)));
      real_type const h = sqrt(r*r-l*l/value_traits::four());

      vector3_type const u = math::unit(x1-x0);
      vector3_type const u_hat = vector3_type(u(1), -u(0), value_traits::zero());

      vector3_type const midpoint = value_traits::half()*(x1+x0);
      vector3_type const c0 = midpoint + math::sgn(angle)*h*u_hat;

      vector3_type p0 = x0;
      vector3_type const l0 = x0-c0;
      real_type const delta = value_traits::two()*angle/segments;
      ++segments;

      glBegin((wireframe?GL_LINES:GL_TRIANGLES));
      for(size_t n = 1; n < segments; ++n)
      {
        real_type const a = n*delta;
        vector3_type const p = math::Rz(-a)*l0+c0;
        if(!wireframe)
        {
          glVertex3d(midpoint(0), midpoint(1), 0.0);
        }
        glVertex3d(p0(0), p0(1), 0.0);
        glVertex3d(p(0), p(1), 0.0);
        p0 = p;
      }
      glEnd();
    }


    /**
    * Draw Circle Arc (draws a 2D circular arc in the xy-plane, rotates about the z-axis)
    *
    * @param x0, x1     The end points of the arc.
    * @parem radius     The radius of the circlular arc in radians.
    *                   This angle is relative to the base line segment (x1 - x0).
    * @param segments   Amount of line steps between x0 and x1 (the higher the smoother)
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename vector3_type>
    inline void DrawCircleArc(vector3_type const & x0, vector3_type const & x1, typename vector3_type::value_type const & radius, size_t segments, bool wireframe)
    {
      typedef typename vector3_type::value_traits  value_traits;
      typedef typename vector3_type::value_type    real_type;
      using std::fabs;

      // draw straight line if angle is close to -or zero, or if segments won't allow an arc
      bool const invalid = radius < value_traits::one()/4096 || segments < 2;
      if(!wireframe || invalid)
      {
        glBegin(GL_LINES);
        glVertex3d(x0(0), x0(1), 0.0);
        glVertex3d(x1(0), x1(1), 0.0);
        glEnd();
        if(invalid)
        {
          return;
        }
      }

      using std::sin;
      using std::cos;
      using std::asin;
      using std::sqrt;
      using std::fabs;

      vector3_type const m = value_traits::half()*(x1+x0);

      real_type const l = math::length(m-x0);
      real_type const r = math::clamp_min(radius, l);
      real_type const phi = asin(l/r);
      real_type const h = r*cos(phi);

      vector3_type const u = math::unit(x1-x0);
      vector3_type const u_hat = vector3_type(u(1), -u(0), value_traits::zero());
      
      vector3_type const c0 = m + h*u_hat;

      vector3_type p0 = x1;  // if x0 is used remember to negate a in Rz(a).
      vector3_type const l0 = p0-c0;
      real_type const delta = (value_traits::two()*phi)/segments;
      ++segments;

      glBegin((wireframe?GL_LINES:GL_TRIANGLES));
      for(size_t n = 1; n < segments; ++n)
      {
        real_type const a = n*delta;
        vector3_type const p = math::Rz(a)*l0+c0;
        if(!wireframe)
        {
          glVertex3d(m(0), m(1), 0.0);
        }
        glVertex3d(p0(0), p0(1), 0.0);
        glVertex3d(p(0), p(1), 0.0);
        p0 = p;
      }
      glEnd();
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_ARC_H
#endif

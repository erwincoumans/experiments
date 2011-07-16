#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_VECTOR_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_VECTOR_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/math/math_vector3.h>
#include <cmath>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw Vector.
    *
    * @param p    The position at which the tail of the vector should be placed.
    * @param v    The vector itself
    */
    template <typename vector3_type>
    inline void DrawVector( vector3_type const & p, vector3_type const & v , double scale = 1.0, bool draw_pos = true  )
    {
      using std::acos;
      using std::sqrt;
      using std::min;

      typedef typename vector3_type::value_type value_type;

      value_type total_length = sqrt( v * v );
      value_type arrow_cone_height = min( 0.2, total_length * 0.2 );
      value_type shaft_height = total_length - arrow_cone_height;

      value_type shaft_radius      = min( 0.05, total_length * 0.05 ) * scale;
      value_type arrow_cone_radius = 2 * shaft_radius;
      value_type origin_radius     = 2 * arrow_cone_radius;

      GLUquadric * qobj = gluNewQuadric();
      GLint slices;
      GLint stacks;

      if (draw_pos) 
      {
        glPushMatrix();
        glTranslatef( p(0), p(1), p(2) );
        slices = 8;
        stacks = 8;
        gluSphere( qobj, origin_radius, slices, stacks );
        glPopMatrix();
      }

      GLfloat angle = 0;
      GLfloat x = 1;
      GLfloat y = 0;
      GLfloat z = 0;

      //--- Compute orientation of normal
      if ( ( v(0) == 0 ) && ( v(1) == 0 ) )
      {
        if ( v(2) > 0 )
        {
          angle = 0;
        }
        else
        {
          angle = 180;
        }
      }
      else
      {
        vector3_type k(0,0,1);
        vector3_type v_unit;
        v_unit = unit( v );
        vector3_type axis = unit( ( v % k ) );
        angle = acos( v_unit * k );
        angle = -180.0 * angle / 3.1415;
        x = axis(0);
        y = axis(1);
        z = axis(2);
      }

      glPushMatrix();
      glTranslatef( p(0), p(1), p(2) );
      glRotatef( angle, x, y, z );
      slices = 12;
      stacks = 1;
      gluCylinder( qobj, shaft_radius, shaft_radius, shaft_height, slices, stacks );
      glPopMatrix();

      glPushMatrix();
      glTranslatef( p(0), p(1), p(2) );
      glRotatef( angle, x, y, z );
      glTranslatef( 0, 0, shaft_height );
      gluCylinder( qobj, arrow_cone_radius, 0, arrow_cone_height, slices, stacks );
      glPopMatrix();

      gluDeleteQuadric( qobj );
    }

    /**
    * Draw Vector.
    *
    * @param v    The vector to be drawn, tail will be placed at origin.
    */
    template <typename vector3_type>
    inline void DrawVector(  vector3_type const & v )
    {
      DrawVector( vector3_type( 0, 0, 0 ), v );
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_VECTOR_H
#endif

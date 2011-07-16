#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_CYLINDER_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_CYLINDER_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <cmath>

namespace OpenTissue
{

  namespace gl
  {
    
    /**
    * Drawing Routine
    *
    * @param cylinder    A reference to the cylinder that should be drawn.
    * @param wireframe   Draw in wireframe or normal.
    */
    template<typename cylinder_type>
    inline void DrawCylinder(cylinder_type const & cylinder, bool wireframe = false) 
    {
      using std::acos;

      typedef typename cylinder_type::real_type     real_type;
      typedef typename cylinder_type::vector3_type  vector3_type;

      vector3_type /*const &*/ center = cylinder.center();
      vector3_type const & axis = cylinder.axis();
      real_type const & radius = cylinder.radius();
      real_type const & height = cylinder.height();

      glPolygonMode(GL_FRONT_AND_BACK,(wireframe?GL_LINE:GL_FILL));
      GLUquadricObj* qobj = gluNewQuadric();
      glPushMatrix();

      glTranslatef(
        (GLfloat)center[0],
        (GLfloat)center[1],
        (GLfloat)center[2]
      );

      GLfloat angle = 0;
      GLfloat x = 1;
      GLfloat y = 0;
      GLfloat z = 0;

      //--- Compute orientation of normal
      // TODO: Comparing floats with == or != is not safe
      if((axis[0]==0) && (axis[1]==0))
      {
        if(axis[2]>0)
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

        vector3_type tmp;
        tmp = axis % k;
        angle = acos(axis * k);
        angle = -180.0*angle/3.1415;
        x = tmp[0];
        y = tmp[1];
        z = tmp[2];
      }

      glRotatef(angle,x,y,z);
      glTranslatef((GLfloat)0,(GLfloat)0,(GLfloat)-height/2.);
      gluCylinder(qobj,radius,radius,height,32,1);
      glPopMatrix();
      gluDeleteQuadric(qobj);
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_CYLINDER_H
#endif

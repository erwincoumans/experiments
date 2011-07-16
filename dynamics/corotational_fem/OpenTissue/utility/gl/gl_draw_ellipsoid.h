#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_ELLIPSOID_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_ELLIPSOID_H
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
    * Draw Ellipsoid.
    *
    * @param ellipsoid  A refernece to the ellipsoid that should be drawn
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename ellipsoid_type>
    inline void DrawEllipsoid(ellipsoid_type const & ellipsoid, bool wireframe = false)
    {
      typedef typename ellipsoid_type::real_type     real_type;
      typedef typename ellipsoid_type::vector3_type  vector3_type;

      vector3_type const & center  = ellipsoid.center();
      vector3_type const & axis0   = ellipsoid.axis0();
      vector3_type const & axis1   = ellipsoid.axis1();
      vector3_type const & axis2   = ellipsoid.axis2();
      real_type    const & radius0 = ellipsoid.radius0();
      real_type    const & radius1 = ellipsoid.radius1();
      real_type    const & radius2 = ellipsoid.radius2();

      glPolygonMode(GL_FRONT_AND_BACK,(wireframe?GL_LINE:GL_FILL));
      GLUquadricObj* qobj = gluNewQuadric();
      glPushMatrix();

      glTranslatef(
        (GLfloat)center(0),
        (GLfloat)center(1),
        (GLfloat)center(2)
        );

      int glindex = 0;
      GLfloat glmatrix[16];
      for (unsigned int col = 0; col < 4; ++col)
      {
        for (unsigned int row = 0; row < 4; ++row)
        {
          if(row==col && row==3)
            glmatrix[glindex++] = 1;
          else if(row==3 || col==3)
            glmatrix[glindex++] = 0;
          else if (col==0)
            glmatrix[glindex++] = axis0(row);
          else if (col==1)
            glmatrix[glindex++] = axis1(row);
          else if (col==2)
            glmatrix[glindex++] = axis2(row);
        }
      }
      glMultMatrixf(glmatrix);

      glScalef(
        (GLfloat)radius0,
        (GLfloat)radius1,
        (GLfloat)radius2
        );

      gluSphere(qobj,1.0,32,32);
      glPopMatrix();
      gluDeleteQuadric(qobj);
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_ELLIPSOID_H
#endif

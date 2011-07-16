#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_OBB_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_OBB_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <OpenTissue/core/math/math_rotation.h>
#include <OpenTissue/core/math/math_constants.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Drawing Routine
    *
    * @param obb
    * @param wireframe   Draw in wireframe or normal.
    */
    template<typename obb_type>
    inline void DrawOBB(obb_type const & obb, bool wireframe = false)
    {
      typedef typename obb_type::real_type       real_type;
      typedef typename obb_type::vector3_type    vector3_type;
      typedef typename obb_type::matrix3x3_type  matrix3x3_type;

      vector3_type /*const &*/ T = obb.center();
      vector3_type const & ext = obb.ext();
      matrix3x3_type const & R = obb.orientation();

      glPushMatrix();
      glTranslatef(
        (GLfloat)T[0],
        (GLfloat)T[1],
        (GLfloat)T[2]
      );
      math::Rotation<real_type>  rot;
      rot = R;
      real_type degrees = math::to_degrees(rot.angle());
      glRotatef((GLfloat)degrees,(GLfloat)rot.axis()[0],(GLfloat)rot.axis()[1],(GLfloat)rot.axis()[2]);
      GLenum const mode = wireframe ? GL_LINE_LOOP : GL_POLYGON;
      //--- pos z
      glBegin(mode);
      glNormal3f(0,0,1);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glNormal3f(0,0,1);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glNormal3f(0,0,1);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glNormal3f(0,0,1);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glEnd();
      //--- pos x
      glBegin(mode);
      glNormal3f(1,0,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glNormal3f(1,0,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glNormal3f(1,0,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glNormal3f(1,0,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glEnd();
      //---- pos y
      glBegin(mode);
      glNormal3f(0,1,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glNormal3f(0,1,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glNormal3f(0,1,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,1,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glEnd();
      //--- neg z
      glBegin(mode);
      glNormal3f(0,0,-1);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,0,-1);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,0,-1);
      glVertex3f((GLfloat)ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,0,-1);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glEnd();
      //--- neg y
      glBegin(mode);
      glNormal3f(0,-1,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,-1,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glNormal3f(0,-1,0);
      glVertex3f((GLfloat)ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glNormal3f(0,-1,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glEnd();
      //--- neg x
      glBegin(mode);
      glNormal3f(-1,0,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)-ext[2]);
      glNormal3f(-1,0,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)-ext[1],(GLfloat)ext[2]);
      glNormal3f(-1,0,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)ext[2]);
      glNormal3f(-1,0,0);
      glVertex3f((GLfloat)-ext[0],(GLfloat)ext[1],(GLfloat)-ext[2]);
      glEnd();
      glPopMatrix();
    }

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_DRAW_OBB_H
#endif

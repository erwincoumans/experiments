#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_AABB_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_AABB_H
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
    * Draw AABB
    * This method draws the AABB in the world coordinate system.
    *
    * @param aabb       A reference to the AABB that should be drawn.
    * @param wireframe  Draw in wireframe or normal.
    */
    template<typename aabb_type>
    inline void DrawAABB(aabb_type const & aabb, bool wireframe = false) 
    {
      GLfloat min_x = aabb.x();
      GLfloat min_y = aabb.y();
      GLfloat min_z = aabb.z();
      GLfloat max_x = min_x + aabb.w();
      GLfloat max_y = min_y + aabb.h();
      GLfloat max_z = min_z + aabb.d();

      GLenum const mode = wireframe ? GL_LINE_LOOP : GL_POLYGON;

      //--- pos z
      glBegin(mode);
      glNormal3f(0,0,1);      glVertex3f(min_x,min_y,max_z);
      glNormal3f(0,0,1);      glVertex3f(max_x,min_y,max_z);
      glNormal3f(0,0,1);      glVertex3f(max_x,max_y,max_z);
      glNormal3f(0,0,1);      glVertex3f(min_x,max_y,max_z);
      glEnd();

      //--- pos x
      glBegin(mode);
      glNormal3f(1,0,0);      glVertex3f(max_x,min_y,max_z);
      glNormal3f(1,0,0);      glVertex3f(max_x,min_y,min_z);
      glNormal3f(1,0,0);      glVertex3f(max_x,max_y,min_z);
      glNormal3f(1,0,0);      glVertex3f(max_x,max_y,max_z);
      glEnd();

      //---- pos y
      glBegin(mode);
      glNormal3f(0,1,0);      glVertex3f(min_x,max_y,max_z);
      glNormal3f(0,1,0);      glVertex3f(max_x,max_y,max_z);
      glNormal3f(0,1,0);      glVertex3f(max_x,max_y,min_z);
      glNormal3f(0,1,0);      glVertex3f(min_x,max_y,min_z);
      glEnd();

      //--- neg z
      glBegin(mode);
      glNormal3f(0,0,-1);      glVertex3f(min_x,min_y,min_z);
      glNormal3f(0,0,-1);      glVertex3f(min_x,max_y,min_z);
      glNormal3f(0,0,-1);      glVertex3f(max_x,max_y,min_z);
      glNormal3f(0,0,-1);      glVertex3f(max_x,min_y,min_z);
      glEnd();

      //--- neg y
      glBegin(mode);
      glNormal3f(0,-1,0);      glVertex3f(min_x,min_y,min_z);
      glNormal3f(0,-1,0);      glVertex3f(max_x,min_y,min_z);
      glNormal3f(0,-1,0);      glVertex3f(max_x,min_y,max_z);
      glNormal3f(0,-1,0);      glVertex3f(min_x,min_y,max_z);
      glEnd();

      //--- neg x
      glBegin(mode);
      glNormal3f(-1,0,0);      glVertex3f(min_x,min_y,min_z);
      glNormal3f(-1,0,0);      glVertex3f(min_x,min_y,max_z);
      glNormal3f(-1,0,0);      glVertex3f(min_x,max_y,max_z);
      glNormal3f(-1,0,0);      glVertex3f(min_x,max_y,min_z);
      glEnd();
    };

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_AABB_H
#endif

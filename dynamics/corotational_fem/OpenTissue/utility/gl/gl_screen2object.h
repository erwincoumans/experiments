#ifndef OPENTISSUE_UTILITY_GL_GL_SCREEN_2_OBJECT_H
#define OPENTISSUE_UTILITY_GL_GL_SCREEN_2_OBJECT_H
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
    * Screen to Object conversion.
    * Given a point on the screen, get the corresponding
    * 3D point in the world
    *
    * @param sx   Screen x-coordinate.
    * @param sy   Screen y-coordinate.
    * @param ox   Object world x-coordinate.
    * @param oy   Object world y-coordinate.
    * @param oz   Object world z-coordinate.
    */
    inline void Screen2Object( int sx, int sy, double & ox, double & oy, double & oz )
    {
      GLdouble projMatrix[ 16 ];
      GLdouble modelViewMatrix[ 16 ];
      GLint viewPort[ 4 ];

      glGetDoublev( GL_MODELVIEW_MATRIX, modelViewMatrix );
      glGetDoublev( GL_PROJECTION_MATRIX, projMatrix );
      glGetIntegerv( GL_VIEWPORT, viewPort );

      GLfloat wx = sx;
      GLfloat wy = viewPort[ 3 ] - sy; // flip y-axe?
      GLfloat wz = 0;

      //--- get z-value at screen position sx,sy
      glReadPixels( static_cast<int>( wx ), static_cast<int>( wy ), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &wz );

      gluUnProject(
        wx, wy, wz
        , modelViewMatrix, projMatrix, viewPort
        , &ox, &oy, &oz
        );
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_SCREEN_2_OBJECT_H
#endif

#ifndef OPENTISSUE_UTILITY_GL_GL_PICKING_H
#define OPENTISSUE_UTILITY_GL_GL_PICKING_H
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
    * Picking Functor.
    */
    class Picking
    {
    protected:

      /**
      * Auxiliary method used by the pick-method.
      *
      * Processes hits, in order to find the one in front of all the others.
      */
      GLuint inFront( GLint hits, GLuint buffer[] )
      {
        GLuint depth = ( GLuint ) - 1;
        GLuint* p = buffer;
        GLuint picked = 0;
        for ( GLint i = 0;i < hits;++i )
        {
          GLboolean save = GL_FALSE;
          GLuint num_names = *p;      /* number of names in this hit */
          p++;

          if ( *p <= depth )   /* check the 1st depth value */
          {
            depth = *p;
            save = GL_TRUE;
          }
          p++;
          if ( *p <= depth )   /* check the 2nd depth value */
          {
            depth = *p;
            save = GL_TRUE;
          }
          p++;

          if ( save )
            picked = *p;

          p += num_names;     /* skip over the rest of the names */
        }
        return picked;
      }

    public:

      /**
      * Get Picked Object Name.
      * The pick method returns a name, during rendering the caller must set this name (by
      * using glPushName and glPopName) for the object being rendered.
      *
      * By default the name value: 0 is used as background, user should not use
      * this value, when setting the object names.
      *
      *
      * @param x       Screen x-coordinates
      * @param y       Screen y-coordinates
      * @param draw    A function pointer to a drawing routine, this method should NOT try
      *                to change model view or projection matrices, nor the viewport, it
      *                should only draw things!!!
      * @return        The name (a GLuint identifier value) of the front most primitive lying at screen location x,y.
      */
      template< typename draw_functor >
      GLuint operator() ( int x, int y, draw_functor const & draw )
      {
        GLdouble projMatrix[ 16 ];
        GLdouble modelViewMatrix[ 16 ];
        GLint viewport[ 4 ];
        GLint mode;

        GLsizei size = 512;
        GLuint buffer[ 512 ];

        glGetIntegerv( GL_MATRIX_MODE, &mode );
        glGetDoublev( GL_MODELVIEW_MATRIX, modelViewMatrix );
        glGetDoublev( GL_PROJECTION_MATRIX, projMatrix );
        glGetIntegerv( GL_VIEWPORT, viewport );

        glSelectBuffer( size, buffer );

        glRenderMode( GL_SELECT );
        glInitNames();

        // setup projection matrix
        glMatrixMode( GL_PROJECTION );
        glPushMatrix();        
        glLoadIdentity();
        // create 5x5 pixel picking region near cursor location 
        gluPickMatrix( ( GLdouble ) x, ( GLdouble ) ( viewport[ 3 ] - y ), 5.0, 5.0, viewport );
        glMultMatrixd( projMatrix );
        
        // setup modelview matrix
        glMatrixMode( GL_MODELVIEW );
        glPushMatrix();      
        glLoadIdentity();
        glMultMatrixd( modelViewMatrix );

        //--- draw stuff ...
        draw();

        glFlush();
        glPopMatrix();        
        glMatrixMode( GL_PROJECTION );
        glPopMatrix();

        GLint hits = glRenderMode( GL_RENDER );
        GLuint picked = inFront( hits, buffer );

        glMatrixMode( mode );

        return picked;
      }
    };

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_PICKING_H
#endif

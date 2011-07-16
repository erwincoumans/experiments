#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_TEXTURE2D_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_TEXTURE2D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * Draw 2D Texture.
    * This function is great for debugging 2D textures. It draws
    * a quad with the texture mapped onto it, using simple brute
    * openGL.
    *
    * @param texture   The texture that should be drawn. 
    */
    template<typename texture_type>
    inline void DrawTexture2D(texture_type const & texture )
    {
      gl::gl_check_errors("DrawTexture2D() - start");
      bool rectangular = texture.width()!=texture.height();
      texture.bind();
      gl::gl_check_errors("DrawTexture2D() - bind");
      if(rectangular)
      {
        glBegin(GL_POLYGON);
        glTexCoord2f(0,0);
        glVertex3d(0,0,0);
        glTexCoord2f(texture.width(),0);
        glVertex3d(texture.width(),0,0);
        glTexCoord2f(texture.width(),texture.height());
        glVertex3d(texture.width(),texture.height(),0);
        glTexCoord2f(0,texture.height());
        glVertex3d(0,texture.height(),0);
        glEnd();
        gl::gl_check_errors("DrawTexture2D() - draw rectangle");
      }
      else
      {
        glBegin(GL_POLYGON);
        glTexCoord2f(0,0);
        glVertex3d(0,0,0);
        glTexCoord2f(1,0);
        glVertex3d(texture.width(),0,0);
        glTexCoord2f(1,1);
        glVertex3d(texture.width(),texture.height(),0);
        glTexCoord2f(0,1);
        glVertex3d(0,texture.height(),0);
        glEnd();
        gl::gl_check_errors("DrawTexture2D() - draw");
      }
      //texture->unbind()....
      gl::gl_check_errors("DrawTexture2D() - done");
    }

  } // namespace gl


} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_TEXTURE2D_H
#endif

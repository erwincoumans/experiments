#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_IMAGE_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_IMAGE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/gpu/texture/texture_texture2D.h>

namespace OpenTissue
{

  namespace gl
  {

    template<typename image_type>
    inline void DrawImage(image_type const & image )
    {
      gl_check_errors("DrawImage() - start");
      bool rectangular = image.width()==image.height();
      OpenTissue::texture::texture2D_pointer texture = image.create_texture(GL_RGBA,rectangular);
      texture->bind();
      if(rectangular)
      {
        glBegin(GL_POLYGON);
        glTexCoord2f(0,0);
        glVertex3d(0,0,0);
        glTexCoord2f(image.width(),0);
        glVertex3d(image.width(),0,0);
        glTexCoord2f(image.width(),image.height());
        glVertex3d(image.width(),image.height(),0);
        glTexCoord2f(0,image.height());
        glVertex3d(0,image.height(),0);
        glEnd();
      }
      else
      {
        glBegin(GL_POLYGON);
        glTexCoord2f(0,0);
        glVertex3d(0,0,0);
        glTexCoord2f(1,0);
        glVertex3d(image.width(),0,0);
        glTexCoord2f(1,1);
        glVertex3d(image.width(),image.height(),0);
        glTexCoord2f(0,1);
        glVertex3d(0,image.height(),0);
        glEnd();
      }
      gl_check_errors("DrawImage() - done");
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_IMAGE_H
#endif

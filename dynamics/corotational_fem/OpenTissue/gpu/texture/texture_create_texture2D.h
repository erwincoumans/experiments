#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE2D_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE2D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/gpu/texture/texture_texture2D.h>
#include <OpenTissue/gpu/texture/gl_is_float_texture_supported.h>

namespace OpenTissue
{
  namespace texture
  {
    /**
    * Create 2D Float Texture.
    *
    * @param width
    * @param height
    * @param components
    * @param pixels
    *
    * @return
    */
    inline texture2D_pointer create_float_texture(
      unsigned int width
      , unsigned int height
      , unsigned int components
      , float const * pixels
      )
    {
      GLenum internal_format;
      GLenum external_format;
      if (components == 1)
      {
        internal_format = GL_FLOAT_R32_NV;
        external_format = GL_RED;
      }
      else if (components == 2)
      {
        //internal_format = GL_LUMINANCE_ALPHA32F_ARB;
        //external_format = GL_LUMINANCE_ALPHA;
        internal_format = GL_FLOAT_RG32_NV;
        external_format = GL_RED;
      }
      else if (components == 3)
      {

        internal_format = GL_FLOAT_RGB32_NV;//GL_RGB32F_ARB;
        external_format = GL_RGB;
      }
      else if (components == 4)
      {
        internal_format = GL_FLOAT_RGBA32_NV;//GL_RGBA32F_ARB;
        external_format = GL_RGBA;
      }
      else
      {
        std::cerr << "create_float_texture(): invalid number of components" << std::endl;
        return texture2D_pointer();
      }
      texture2D_pointer tex;
      tex.reset( new Texture2D ( internal_format, width, height, external_format, GL_FLOAT, pixels ) );
      tex->bind();
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      return tex;
    }

    /**
    * Create 2D Float Texture RECT.
    *
    *
    *
    * @param width
    * @param height
    * @param components
    * @param pixels
    *
    * @return
    */
    inline texture2D_pointer create_float_texture_rectangle(
      unsigned int width
      , unsigned int height
      , unsigned int components
      , float const * pixels
      )
    {
      GLenum internal_format;
      GLenum external_format;
      if (components == 1)
      {
        internal_format = GL_FLOAT_R32_NV;
        external_format = GL_RED;
      }
      else if (components == 2)
      {
        internal_format = GL_FLOAT_RG32_NV;
        external_format = GL_RED;
      }
      else if (components == 3)
      {

        internal_format = GL_FLOAT_RGB32_NV;//GL_RGB32F_ARB;
        external_format = GL_RGB;
      }
      else if (components == 4)
      {
        internal_format = GL_FLOAT_RGBA32_NV;//GL_RGBA32F_ARB;
        external_format = GL_RGBA;
      }
      else
      {
        std::cerr << "create_float_texture_rectangle(): invalid number of components" << std::endl;
        return texture2D_pointer();
      }
      texture2D_pointer tex;
      tex.reset(
        new Texture2D ( internal_format, width, height, external_format, GL_FLOAT, pixels, true )
        );
      return tex;
    }

    /**
    * Create Unsigned Byte 2D Texture.
    *
    * @param width
    * @param height
    * @param components
    * @param pixels
    *
    * @return
    */
    inline texture2D_pointer create_unsigned_byte_texture_rectangle(
      unsigned int width
      , unsigned int height
      , unsigned int components
      , unsigned char const * pixels
      )
    {
      GLenum internal_format;
      GLenum external_format;
      if (components == 3)
      {
        internal_format = GL_RGB8;
        external_format = GL_RGB;
      }
      else if (components == 4)
      {
        internal_format = GL_RGBA8;
        external_format = GL_RGBA;
      }
      else
      {
        std::cerr << "create_unsigned_byte_texture_rectangle(): invalid number of components" << std::endl;
        return texture2D_pointer();
      }
      texture2D_pointer tex;
      tex.reset( new Texture2D ( internal_format, width, height, external_format, GL_UNSIGNED_BYTE, pixels, true ) );
      tex->bind();
      glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      return tex;
    }

    /**
    * Create Unsigned Byte 2D Texture Rect.
    *
    * @param width
    * @param height
    * @param components
    * @param pixels
    *
    * @return
    */
    inline texture2D_pointer create_unsigned_byte_texture(
      unsigned int width
      , unsigned int height
      ,  unsigned int components
      , const unsigned char *pixels
      )
    {
      GLenum internal_format;
      GLenum external_format;
      if (components == 3)
      {
        internal_format = GL_RGB8;
        external_format = GL_RGB;
      }
      else if (components == 4)
      {
        internal_format = GL_RGBA8;
        external_format = GL_RGBA;
      }
      else
      {
        std::cerr << "create_unsigned_byte_texture(): invalid number of components" << std::endl;
        return texture2D_pointer();
      }
      texture2D_pointer tex;
      tex.reset( new Texture2D ( internal_format, width, height, external_format, GL_UNSIGNED_BYTE, pixels ) );
      tex->bind();
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      return tex;
    }

  } // namespace texture

} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_TEXTURE_CREATE_TEXTURE2D_H
#endif

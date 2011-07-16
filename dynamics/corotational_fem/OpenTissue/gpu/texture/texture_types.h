#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_TYPES_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_TYPES_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <cassert>

namespace OpenTissue
{
  namespace texture
  {
    /**
    * Determine External Pixel Format.
    * External refers to CPU side, whereas internal refers to GPU side.
    *
    * @param channels   The number of channels in the external pixel data.
    * @return           The best openGL type for the external format.
    */
    inline unsigned int external_format( unsigned int channels )
    {
      switch ( channels )
      {
      case 4:
        return GL_RGBA;
      case 3:
        return GL_RGB;
      case 2:
        return GL_LUMINANCE_ALPHA;
      case 1:
        return GL_LUMINANCE;
      default:
        assert(!"external_format(): OpenGL only support up to 4 channels." );
      }
      return GL_INVALID_ENUM;
    }

    /**
    * Determine External Pixel Type.
    * External refers to CPU side, whereas internal refers to GPU side.
    * This function is implemented using specialization, Example usage:
    *
    *   unsigned int type = external_type<unsigned char>();
    *
    * @return           The best openGL type for the external type.
    */
    template<typename T>
    inline unsigned int external_type();

    template<>
    inline unsigned int external_type<double>() {    return GL_DOUBLE; }

    template<>
    inline unsigned int external_type<float>() {    return GL_FLOAT; }

    template<>
    inline unsigned int external_type<int>() {    return GL_INT; }

    template<>
    inline unsigned int external_type<unsigned int>() {    return GL_UNSIGNED_INT; }

    template<>
    inline unsigned int external_type<unsigned short>() {    return GL_UNSIGNED_SHORT; }

    template<>
    inline unsigned int external_type<short>() {    return GL_SHORT; }

    template<>
    inline unsigned int external_type<char>() {    return GL_UNSIGNED_BYTE; }

    template<>
    inline unsigned int external_type<unsigned char>() { return GL_UNSIGNED_BYTE; }


    /**
    * Determine Internal Pixel Type.
    * External refers to CPU side, whereas internal refers to GPU side.
    * This function is implemented using specialization, Example usage:
    *
    *   unsigned int format = internal_format<unsigned char>(4);
    *
    * The specialization types indicates the number of bits in each color
    * component (or if a float texture is wanted). The channels
    * parameter indicates how many color components that is wanted.
    *
    * @param channels   The number of channels in the internal pixel format.
    * @return           The best openGL type for the internal format.
    */
    template<typename T>
    inline unsigned int internal_format(int channels);


    template<>
    inline unsigned int internal_format<unsigned char>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_INTENSITY8;
      case 2:  return GL_LUMINANCE8_ALPHA8;
      case 3:  return GL_RGB8;
      case 4:  return GL_RGBA8;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline unsigned int internal_format<char>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_INTENSITY8;
      case 2:  return GL_LUMINANCE8_ALPHA8;
      case 3:  return GL_RGB8;
      case 4:  return GL_RGBA8;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline unsigned int internal_format<short>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_INTENSITY16;
      case 2:  return GL_LUMINANCE16_ALPHA16;
      case 3:  return GL_RGB16;
      case 4:  return GL_RGBA16;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline unsigned int internal_format<unsigned short>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_INTENSITY16;
      case 2:  return GL_LUMINANCE16_ALPHA16;
      case 3:  return GL_RGB16;
      case 4:  return GL_RGBA16;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline unsigned int internal_format<int>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_FLOAT_R32_NV;
      case 2:  return GL_LUMINANCE_ALPHA;
      case 3:  return GL_RGB;
      case 4:  return GL_RGBA;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline unsigned int internal_format<unsigned int>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_FLOAT_R32_NV;
      case 2:  return GL_LUMINANCE_ALPHA;
      case 3:  return GL_RGB;
      case 4:  return GL_RGBA;
      }
      return GL_INVALID_ENUM;
    }

    template<>
    inline  unsigned int internal_format<float>(int channels)
    {
      switch(channels)
      {
      case 1:  return GL_FLOAT_R32_NV;
      case 2:  return GL_LUMINANCE_ALPHA;
      case 3:  return GL_RGB;
      case 4:  return GL_RGBA;
      }
      return GL_INVALID_ENUM;
    }

  } // namespace texture

} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_TEXTURE_TYPES_H
#endif

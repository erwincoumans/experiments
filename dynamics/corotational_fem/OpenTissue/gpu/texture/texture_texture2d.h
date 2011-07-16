#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE2D_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE2D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_power2.h>
#include <OpenTissue/gpu/texture/texture_types.h>
#include <OpenTissue/utility/gl/gl_check_errors.h>


#include <iostream>
#include <cassert>

namespace OpenTissue
{
  namespace texture
  {

    /**
    * A 2D Texture convenience Class.
    * This class encapsylates all openGL technicalities in dealing with 2D textures.
    */
    class Texture2D
    {
    protected:

      GLuint       m_texture_ID;          ///< The ID of the texture object.
      int          m_internal_format;     ///< The internal format of the texture, i.e. how many color compoents (and their size).
      bool         m_owned;               ///< Boolean flag indicating whether this texture instance owns the texture object.
      int          m_width;               ///< The width In pixels of texture.
      int          m_height;              ///< The height In pixels of texture.
      unsigned int m_format;              ///< External pixel format, i.e how many components.
      unsigned int m_type;                ///< External pixel type, i.e. unsigned char, unsigned short etc..
      bool         m_rectangular;         ///< If true texture coordiates are in the range [0..width]x[0..height] otherwise they are in the range [0..1]x[0..1]
      unsigned int m_texture_target;      ///< The texture target.

    public:

      unsigned int get_texture_ID()      const { return m_texture_ID;      }
      unsigned int get_texture_target()  const { return m_texture_target;  }
      int          width()               const { return m_width;           }
      int          height()              const { return m_height;          }
      int          get_internal_format() const { return m_internal_format; }
      bool         rectangular()         const { return m_rectangular;     }

    protected:

      void create(unsigned int texture_ID = 0)
      {
        m_texture_ID = texture_ID;
        m_owned=false;
        if (m_texture_ID==0)
        {
          glGenTextures(1, &m_texture_ID);
          m_owned = true;
          if(m_texture_ID == 0)
          {
            std::cerr<< "Texture2D::create(): Could not create texture ID";
          }
        }
      }

    public:

      /**
      * Texture2D Constructor.
      *
      * @param internal_format  The number of color components in the texture.
      *                         Must be 1, 2, 3, or 4, or one of the following
      *                         symbolic constants: GL_ALPHA, GL_ALPHA4, GL_ALPHA8,
      *                         GL_ALPHA12, GL_ALPHA16, GL_LUMINANCE, GL_LUMINANCE4,
      *                         GL_LUMINANCE8, GL_LUMINANCE12, GL_LUMINANCE16,
      *                         GL_LUMINANCE_ALPHA, GL_LUMINANCE4_ALPHA4, GL_LUMINANCE6_ALPHA2,
      *                         GL_LUMINANCE8_ALPHA8, GL_LUMINANCE12_ALPHA4, GL_LUMINANCE12_ALPHA12,
      *                         GL_LUMINANCE16_ALPHA16, GL_INTENSITY, GL_INTENSITY4, GL_INTENSITY8,
      *                         GL_INTENSITY12, GL_INTENSITY16, GL_R3_G3_B2, GL_RGB, GL_RGB4, GL_RGB5,
      *                         GL_RGB8, GL_RGB10, GL_RGB12, GL_RGB16, GL_RGBA, GL_RGBA2, GL_RGBA4,
      *                         GL_RGB5_A1, GL_RGBA8, GL_RGB10_A2, GL_RGBA12, or GL_RGBA16.
      * @param width            The width of the texture image.
      * @param height           The height of the texture image.
      * @param format           The format of the pixel data. It can assume one of the symbolic values:
      *                         GL_COLOR_INDEX, GL_RED, GL_GREEN,GL_BLUE,GL_ALPHA,GL_RGB,GL_RGBA,GL_BGR_EXT,GL_BGR_EXT
      *                         GL_BGRA_EXT,GL_LUMINANCE,GL_LUMINANCE_ALPHA
      * @param type             The data type of the pixel data. The following symbolic values are accepted:
      *                         GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT,
      *                         GL_INT, and GL_FLOAT.
      * @param pixels           A pointer to the image data in memory.
      *
      * @param rectangular     Set to true if a rectangular texture is wanted, default is false.
      * @param border          Set border value, default is zero
      */
      Texture2D(
        int internal_format
        , int width
        , int height
        , int format
        , int type
        , const void *pixels
        , bool rectangular=false
        , int border=0
        )
        : m_internal_format(internal_format)
        , m_width(width)
        , m_height(height)
        , m_format(format)
        , m_type(type)
        , m_rectangular(rectangular)
        , m_texture_target( rectangular?GL_TEXTURE_RECTANGLE_ARB:GL_TEXTURE_2D )
      {
        glPushAttrib(GL_TEXTURE_BIT /*| GL_HINT_BIT | GL_ENABLE_BIT*/ );

        assert((m_rectangular || math::is_power2( m_width  )) || !"Texture2D(): Width was not a power of two");
        assert((m_rectangular || math::is_power2( m_height )) || !"Texture2D(): Height was not a power of two");
        create();
        if(!glIsEnabled( m_texture_target ) )
          glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);

        glTexImage2D(m_texture_target, 0, internal_format, width, height,  border, format, (GLenum) type, pixels);

        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri( m_texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri( m_texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glPopAttrib();
      }

      Texture2D()
        : m_rectangular(false)
        , m_texture_target( m_rectangular?GL_TEXTURE_RECTANGLE_ARB:GL_TEXTURE_2D )
      {
        create();
        glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);
      }

      Texture2D(int w, int h, bool r=false)
        : m_width(w)
        , m_height(h)
        , m_rectangular(r)
        , m_texture_target( m_rectangular?GL_TEXTURE_RECTANGLE_ARB:GL_TEXTURE_2D )
      {
        create();
        glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);
      }

      ~Texture2D()
      {
        if (m_texture_ID!=0 && m_owned)
          glDeleteTextures(1, &m_texture_ID);
      }

    public:

      /**
      * Bind Texture.
      * This method makes sure that the texture target is enabled and the
      * texture object is bound.
      */
      void bind() const
      {
        //--- Make sure that rectangle mode and texture 2D mode are not enabled at the same time!
        if(m_texture_target==GL_TEXTURE_2D)
        {
          if(glIsEnabled(GL_TEXTURE_RECTANGLE_ARB))
            glDisable(GL_TEXTURE_RECTANGLE_ARB);
          if(glIsEnabled(GL_TEXTURE_RECTANGLE_NV))
            glDisable(GL_TEXTURE_RECTANGLE_NV);
        }
        else
        {
          if(glIsEnabled(GL_TEXTURE_2D))
            glDisable(GL_TEXTURE_2D);
        }
        if( !glIsEnabled(m_texture_target) )
          glEnable(m_texture_target);

        glBindTexture(m_texture_target, m_texture_ID);
      }

      /**
      * Load Texture From Memory.
      *
      * @param pixels   A pointer to the contiguos memory location where texture
      *                 data should be loaded from.
      */
      template<typename T>
      void load(T const * pixels)
      {
        assert(pixels != 0 || !"Texture2D::load() - pixels are null");

        //unsigned int ext_channels = 1;
        //m_format = external_format( ext_channels );
        //m_type   = external_type<T>();

        bind();
        glTexSubImage2D(
          m_texture_target //GLenum target
          , 0              //GLint level
          , 0              //GLint xoffset
          , 0              //GLint yoffset
          , m_width        //GLsizei width
          , m_height       //GLsizei height
          , m_format       //GLenum format
          , m_type         //GLenum type
          , pixels         //const GLvoid *pixels
          );
      }

      /**
      * Set Texture Wrapping Mode.
      *
      * @param flag   If true texture wrapping is set to GL_REPEAT otherwise it is set to GL_CLAMP.
      */
      void set_repeating(bool flag)
      {
        bind();
        if(flag)
        {
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_REPEAT);
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_REPEAT);
        }
        else
        {
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP);
        }
      }

      void set_mipmapping(bool flag)
      {
        assert( (flag && (m_texture_target == GL_TEXTURE_2D)) || !"Texture2D::set_mipmapping(): Mipmap generation only supported on GL_TEXTURE_2D texture targets");

        bind();

        if(flag)
        {
          glTexParameteri( m_texture_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);        
          glTexParameteri( m_texture_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
          OpenTissue::gl::gl_check_errors("GL_TEXTURE_MIN_FILTER = GL_LINEAR_MIPMAP_LINEAR");

          glTexParameterf( m_texture_target, GL_GENERATE_MIPMAP, GL_TRUE);  // Support for dynamic textures
          OpenTissue::gl::gl_check_errors("GL_GENERATE_MIPMAP = TRUE");

          glHint( GL_GENERATE_MIPMAP_HINT, GL_NICEST );
          OpenTissue::gl::gl_check_errors("GL_GENERATE_MIPMAP_HINT = GL_NICEST");

          glGenerateMipmapEXT( m_texture_target );   // New way of generating mipmap levels
          OpenTissue::gl::gl_check_errors("glGenerateMipmapEXT");
        }
        else
        {
          glTexParameteri( m_texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTexParameteri( m_texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
      }

      // convenience methods

#ifdef MACOS
      void parameter(GLenum pname, int i)
      {
        glTexParameteri(m_texture_target, pname, i);
      }
#endif

      void parameter(GLenum pname, GLint i)
      {
        glTexParameteri(m_texture_target, pname, i);
      }

      void parameter(GLenum pname, GLfloat f)
      {
        glTexParameterf(m_texture_target, pname, f);
      }

      void parameter(GLenum pname, GLint * ip)
      {
        glTexParameteriv(m_texture_target, pname, ip);
      }

      void parameter(GLenum pname, GLfloat * fp)
      {
        glTexParameterfv(m_texture_target, pname, fp);
      }


      /**
      * Validate Sampler Type.
      * This method tests if sampler type is compatible with this 2D texture.
      * This method is invoked by Program when setting an input
      * texture for a Cg program.
      *
      * @param sampler_type   A textutual representation of the Cg sampler type.
      *
      * @return  If the sampler type is valid the return value is true other wise it is false.
      */
      bool is_valid_sampler_type(std::string const & sampler_type) const
      {
        if (m_rectangular)
        {
          if (sampler_type!= "sampler2DRect" && sampler_type!="samplerRECT")
          {
            std::cerr << "Texture2D::validate_sampler_type(): TextureRectangle attempted to bound to sampler type " << sampler_type << std::endl;
            return false;
          }
        }
        else
        {
          if (sampler_type!="sampler2D")
          {
            std::cerr << "Texture2D::validate_sampler_type(): Attempted to bound to sampler type " << sampler_type << std::endl;
            return false;
          }
        }
        return true;
      }

    };

    typedef Texture2D*  texture2D_pointer;

  } // namespace texture
} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE2D_H
#endif

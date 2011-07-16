#ifndef OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE3D_H
#define OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE3D_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/core/math/math_power2.h>
#include <OpenTissue/gpu/texture/texture_types.h>


#include <iostream>
#include <cassert>

namespace OpenTissue
{
  namespace texture
  {

    /**
    * A 3D Texture convenience Class.
    * This class encapsylates all openGL technicalities in dealing with 3D textures.
    */
    class Texture3D
    {
    protected:

      // unsigned int m_texture_ID;          ///< The ID of the texture object.
      GLuint       m_texture_ID;          ///< The ID of the texture object.
      int          m_internal_format;     ///< The internal format of the texture, i.e. how many color compoents (and their size).
      bool         m_owned;               ///< Boolean flag indicating whether this texture instance owns the texture object.
      int          m_image_skip_i;        ///< The pixel offset in an image row, from where to start loading pixel data.
      int          m_image_skip_j;        ///< The row offset in an image, from where to start loading pixel data.
      int          m_image_skip_k;        ///< The image offset in an volume, from where to start loading pixel data.
      int          m_image_size_i;        ///< The number of pixels in a image row.
      int          m_image_size_j;        ///< The number of rows in a image.
      int          m_image_size_k;        ///< The number of images in volume.
      int          m_texture_size_i;      ///< The number of pixels in a row of the texture.
      int          m_texture_size_j;      ///< The number of rows of an image of the texture.
      int          m_texture_size_k;      ///< The number of images in the texture.
      unsigned int m_format;              ///< External pixel format, i.e how many components.
      unsigned int m_type;                ///< External pixel type, i.e. unsigned char, unsigned short etc..
      bool         m_cubic;               ///< If true texture coordiates are in the range [0..width]x[0..height]x[0..depth] otherwise they are in the range [0..1]x[0..1]x[0..1]
      unsigned int m_texture_target;      ///< The texture target.
      float        m_unpack_scale;        ///< Scaling of pixel data when transfering from processor to GPU. Default value is 1.0.

    public:

      unsigned int get_texture_ID()      const { return m_texture_ID;      }
      unsigned int get_texture_target()  const { return m_texture_target;  }
      int          width()               const { return m_texture_size_i;  }
      int          height()              const { return m_texture_size_j;  }
      int          depth()               const { return m_texture_size_k;  }
      int          get_internal_format() const { return m_internal_format; }
      bool         cubic()               const { return m_cubic;           }
      float        unpack_scale()        const { return m_unpack_scale;    }

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
            std::cerr<< "Texture3D::create(): Could not create texture ID";
          }
        }
      }

    public:

      /**
      * Get Texture Size
      * This method test is the desired texture can be loaded into texture memory. Note
      * it does not test whether the texture would actually be resident in texture memory. It
      * only tests if there is available resources!!!
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
      * @param depth            The depth of the texture image.
      * @param external_format  The format of the pixel data. It can assume one of the symbolic values:
      *                         GL_COLOR_INDEX, GL_RED, GL_GREEN,GL_BLUE,GL_ALPHA,GL_RGB,GL_RGBA,GL_BGR_EXT,GL_BGR_EXT
      *                         GL_BGRA_EXT,GL_LUMINANCE,GL_LUMINANCE_ALPHA
      * @param external_type    The data type of the pixel data. The following symbolic values are accepted:
      *                         GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT,
      *                         GL_INT, and GL_FLOAT.
      *
      * @return   The texture size in bytes, or zero if the texture can not be loaded into texture memory.
      */
      int check_texture_size(
        int internal_format
        , int width
        , int height
        , int depth
        , int external_format
        , int external_type
        )
      {
        using std::ceil;

        //--- Test texture dimensions against lower bounds on 3D texture size!
        GLint max_tex_size;
        GLint max_tex_size_ext;
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_tex_size);
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE_EXT, &max_tex_size_ext);
        if(width>max_tex_size ||height>max_tex_size || depth>max_tex_size )
          return 0;
        if(width>max_tex_size_ext ||height>max_tex_size_ext || depth>max_tex_size_ext )
          return 0;

        //--- see if texture is capable of being loaded into texture memory
        //---
        //--- If successfull this does not imply that texture will be resident in texture memory!!!
        unsigned int m_proxy_texture_target = GL_PROXY_TEXTURE_3D;
        glTexImage3D( m_proxy_texture_target, 0, internal_format, width, height, depth, 0, external_format, external_type, 0 );
        GLint tex_size[ 3 ] = { 0, 0, 0 };
        GLint channel_size[ 6 ] = { 0, 0, 0, 0, 0, 0};
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_WIDTH         , &( tex_size[ 0 ] )     );
        if(tex_size[0]==0)
          return 0;
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_HEIGHT        , &( tex_size[ 1 ] )     );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_DEPTH         , &( tex_size[ 2 ] )     );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_RED_SIZE      , &( channel_size[ 0 ] ) );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_GREEN_SIZE    , &( channel_size[ 1 ] ) );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_BLUE_SIZE     , &( channel_size[ 2 ] ) );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_ALPHA_SIZE    , &( channel_size[ 3 ] ) );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_LUMINANCE_SIZE, &( channel_size[ 4 ] ) );
        glGetTexLevelParameteriv( m_proxy_texture_target, 0, GL_TEXTURE_INTENSITY_SIZE, &( channel_size[ 5 ] ) );
        int total_bits = channel_size[ 0 ] + channel_size[ 1 ] + channel_size[ 2 ] + channel_size[ 3 ] + channel_size[ 4 ] + channel_size[ 5 ];
        float bytes = static_cast< float >( ceil( total_bits / 8.0 ));
        int memory_size = static_cast<int>(tex_size[ 0 ] * tex_size[ 1 ] * tex_size[ 2 ] * bytes);
        return memory_size;
      }


    protected:

      GLint m_tmp[10];  ///< Temporariy storage used to stack pixel store settings

      /**
      * Sets up Pixel Transfer Specifics.
      * This method must be paired with a call to clear_pixel_transfer.
      */
      void setup_pixel_transfer()
      {
        //--- Example explaining the need for scaling!!!
        //---
        //--- Problem with 12bit dicom stored as 16 bit unsighed short
        //---
        //---      2^16 - 1 = 65535   -> 1.0
        //---
        //--- Four bits are unussed, so we really wanted
        //---
        //---      2^12 - 1 = 4095    -> 1.0
        //---
        //--- Thus if we scale pixels while they are being unpacked, then we can take care of this.
        //---
        //---  scale = 65535/4095 = 16.003663003663003663003663003663 ~ 16
        //---
        //float unpack_scale = 16;
        float bias  = 0.0;

        //std::cout << "Texture3D::setup_pixel_transfer(): unpack scale  = " << m_unpack_scale << std::endl;
        //std::cout << "Texture3D::setup_pixel_transfer(): unpack bias   = " << bias << std::endl;
        glPixelTransferf ( GL_RED_SCALE           , m_unpack_scale );
        glPixelTransferf ( GL_GREEN_SCALE         , m_unpack_scale );
        glPixelTransferf ( GL_BLUE_SCALE          , m_unpack_scale );
        glPixelTransferf ( GL_RED_BIAS            , bias           );
        glPixelTransferf ( GL_GREEN_BIAS          , bias           );
        glPixelTransferf ( GL_BLUE_BIAS           , bias           );
        //---
        //---  These should be the width and height of the CPU side stored image
        //---  and not the widht and height of this texture!!!
        //---
        glGetIntegerv    ( GL_UNPACK_ROW_LENGTH   , &m_tmp[0]  );
        glGetIntegerv    ( GL_UNPACK_IMAGE_HEIGHT , &m_tmp[1]  );
        glGetIntegerv    ( GL_UNPACK_ALIGNMENT    , &m_tmp[2]  );
        glGetIntegerv    ( GL_UNPACK_SKIP_PIXELS  , &m_tmp[3]  );
        glGetIntegerv    ( GL_UNPACK_SKIP_ROWS    , &m_tmp[4]  );
        glGetIntegerv    ( GL_UNPACK_SKIP_IMAGES  , &m_tmp[5]  );

        //std::cout << "Texture3D::setup_pixel_transfer(): pixel skip   = " << m_image_skip_i << std::endl;
        //std::cout << "Texture3D::setup_pixel_transfer(): row skip     = " << m_image_skip_j << std::endl;
        //std::cout << "Texture3D::setup_pixel_transfer(): image skip   = " << m_image_skip_k << std::endl;
        //std::cout << "Texture3D::setup_pixel_transfer(): row length   = " << m_image_size_i << std::endl;
        //std::cout << "Texture3D::setup_pixel_transfer(): image height = " << m_image_size_j << std::endl;

        glPixelStorei    ( GL_UNPACK_ALIGNMENT    , 1                );
        glPixelStorei    ( GL_UNPACK_ROW_LENGTH   , m_image_size_i   );
        glPixelStorei    ( GL_UNPACK_IMAGE_HEIGHT , m_image_size_j   );
        glPixelStorei    ( GL_UNPACK_SKIP_PIXELS  , m_image_skip_i   );
        glPixelStorei    ( GL_UNPACK_SKIP_ROWS    , m_image_skip_j   );
        glPixelStorei    ( GL_UNPACK_SKIP_IMAGES  , m_image_skip_k   );
      }


      /**
      * Clear Pixel Transfer Specifics.
      * This method must should be preceeded by a call to setup_pixel_transfer.
      */
      void clear_pixel_transfer()
      {
        glPixelStorei    ( GL_UNPACK_ROW_LENGTH   , m_tmp[0] );
        glPixelStorei    ( GL_UNPACK_IMAGE_HEIGHT , m_tmp[1] );
        glPixelStorei    ( GL_UNPACK_ALIGNMENT    , m_tmp[2] );
        glPixelStorei    ( GL_UNPACK_SKIP_PIXELS  , m_tmp[3] );
        glPixelStorei    ( GL_UNPACK_SKIP_ROWS    , m_tmp[4] );
        glPixelStorei    ( GL_UNPACK_SKIP_IMAGES  , m_tmp[5] );
        //--- We are lazy, we just restore default values!!! We
        //--- should stack these values:-)
        glPixelTransferf ( GL_RED_SCALE           , 1.0      );
        glPixelTransferf ( GL_GREEN_SCALE         , 1.0      );
        glPixelTransferf ( GL_BLUE_SCALE          , 1.0      );
        glPixelTransferf ( GL_RED_BIAS            , 0        );
        glPixelTransferf ( GL_GREEN_BIAS          , 0        );
        glPixelTransferf ( GL_BLUE_BIAS           , 0        );
      }

    public:

      /**
      * Texture3D Constructor.
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
      * @param image_width      The width of the image.
      * @param image_height     The height of the image.
      * @param image_depth      The depth of the image.
      * @param format           The format of the pixel data. It can assume one of the symbolic values:
      *                         GL_COLOR_INDEX, GL_RED, GL_GREEN,GL_BLUE,GL_ALPHA,GL_RGB,GL_RGBA,GL_BGR_EXT,GL_BGR_EXT
      *                         GL_BGRA_EXT,GL_LUMINANCE,GL_LUMINANCE_ALPHA
      * @param type             The data type of the pixel data. The following symbolic values are accepted:
      *                         GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT,
      *                         GL_INT, and GL_FLOAT.
      * @param pixels           A pointer to the image data in memory.
      *
      * @param cubic
      * @param unpack_scale     Pixel transfering scaling during unpacking. (12 bit Dicom usually needs scale=16)
      *
      */
      Texture3D(
        int internal_format
        , int image_width
        , int image_height
        , int image_depth
        , int format
        , int type
        , const void *pixels
        , bool cubic=false
        , float unpack_scale = 1.0
        )
        : m_internal_format(internal_format)
        , m_image_skip_i(0)
        , m_image_skip_j(0)
        , m_image_skip_k(0)
        , m_image_size_i( image_width  )
        , m_image_size_j( image_height )
        , m_image_size_k( image_depth  )
        , m_texture_size_i( math::lower_power2(m_image_size_i) )
        , m_texture_size_j( math::lower_power2(m_image_size_j) )
        , m_texture_size_k( math::lower_power2(m_image_size_k) )
        , m_format(format)
        , m_type(type)
        , m_cubic(cubic)
        , m_texture_target( GL_TEXTURE_3D )//, m_texture_target( cubic?GL_TEXTURE_RECTANGLE_ARB:GL_TEXTURE_3D )
        , m_unpack_scale(unpack_scale)
      {
        //std::cout << "Texture3D():  texture dimensions = (" << m_texture_size_i << "x" << m_texture_size_j << "x" << m_texture_size_k << ")" << std::endl;
        assert( math::is_power2( m_texture_size_i ) || !"Texture3D(): Texture width was not a power of two"  );
        assert( math::is_power2( m_texture_size_j ) || !"Texture3D(): Texture height was not a power of two" );
        assert( math::is_power2( m_texture_size_k ) || !"Texture3D(): Texture depth was not a power of two"  );
        int memory_size = check_texture_size(internal_format,m_texture_size_i,m_texture_size_j,m_texture_size_k,format,type);
        assert(memory_size>0 || !"Texture3D():  insuficient memory could not load 3D texture");
        std::cout << "Texture3D():  memory = " << memory_size << " bytes" << std::endl;

        create();
        glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);
        setup_pixel_transfer();

        //--- Setup texture parameters
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri( m_texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        //--- Create Texture Object
        glTexImage3D(
          m_texture_target
          , 0
          , internal_format
          , m_texture_size_i
          , m_texture_size_j
          , m_texture_size_k
          , 0
          , format
          , type
          , pixels
          );

        clear_pixel_transfer();
      }

      /**
      * Construct Empty Texture.
      * This constructor allocates a texture of the specified size, without loading any data into the texture.
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
      * @param texture_size_i      The width of the image.
      * @param texture_size_j     The height of the image.
      * @param texture_size_k      The depth of the image.
      * @param format           The format of the pixel data. It can assume one of the symbolic values:
      *                         GL_COLOR_INDEX, GL_RED, GL_GREEN,GL_BLUE,GL_ALPHA,GL_RGB,GL_RGBA,GL_BGR_EXT,GL_BGR_EXT
      *                         GL_BGRA_EXT,GL_LUMINANCE,GL_LUMINANCE_ALPHA
      * @param type             The data type of the pixel data. The following symbolic values are accepted:
      *                         GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT,
      *                         GL_INT, and GL_FLOAT.
      * @param cubic
      * @param unpack_scale     Pixel transfering scaling during unpacking. (12 bit Dicom usually needs scale=16)
      *
      */
      Texture3D(
        int internal_format
        , int texture_size_i
        , int texture_size_j
        , int texture_size_k
        , int format
        , int type
        , bool cubic=false
        , float unpack_scale = 1.0
        )
        : m_internal_format(internal_format)
        , m_image_skip_i(0)
        , m_image_skip_j(0)
        , m_image_skip_k(0)
        , m_image_size_i( texture_size_i  )
        , m_image_size_j( texture_size_j )
        , m_image_size_k( texture_size_k  )
        , m_texture_size_i( math::lower_power2(texture_size_i) )
        , m_texture_size_j( math::lower_power2(texture_size_j) )
        , m_texture_size_k( math::lower_power2(texture_size_k) )
        , m_format(format)
        , m_type(type)
        , m_cubic(cubic)
        , m_texture_target( GL_TEXTURE_3D )//, m_texture_target( cubic?GL_TEXTURE_RECTANGLE_ARB:GL_TEXTURE_3D )
        , m_unpack_scale(unpack_scale)
      {
        //std::cout << "Texture3D():  texture dimensions = (" << m_texture_size_i << "x" << m_texture_size_j << "x" << m_texture_size_k << ")" << std::endl;
        assert( math::is_power2( m_texture_size_i ) || !"Texture3D(): Texture width was not a power of two"  );
        assert( math::is_power2( m_texture_size_j ) || !"Texture3D(): Texture height was not a power of two" );
        assert( math::is_power2( m_texture_size_k ) || !"Texture3D(): Texture depth was not a power of two"  );
        int memory_size = check_texture_size(internal_format,m_texture_size_i,m_texture_size_j,m_texture_size_k,format,type);
        assert(memory_size>0 || !"Texture3D():  insuficient memory could not load 3D texture");
        std::cout << "Texture3D():  memory = " << memory_size << " bytes" << std::endl;
        create();
        glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);
        //--- Setup texture parameters
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri( m_texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri( m_texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        //--- Create Texture Object
        glTexImage3D(
          m_texture_target
          , 0
          , internal_format
          , m_texture_size_i
          , m_texture_size_j
          , m_texture_size_k
          , 0
          , format
          , type
          , 0
          );
      }

      ~Texture3D()
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
        if( !glIsEnabled(m_texture_target) )
          glEnable(m_texture_target);
        glBindTexture(m_texture_target, m_texture_ID);
      }


      /**
      * Load sub region of image into subregion of texture.
      *
      * @param image_skip_i         The image pixel offset from where loading should start.
      * @param image_skip_j         The image row offset from where loading should start.
      * @param image_skip_k         The image offset from where loading should start.
      * @param image_size_i         The total number of pixels along i-axis.
      * @param image_size_j         The total number of pixels along j-axis.
      * @param image_size_k         The total number of pixels along k-axis.
      * @param texture_offset_i     The row starting position.
      * @param texture_offset_j     The height starting position.
      * @param texture_offset_k     The image starting position.
      * @param texture_fill_i       The number of pixels to fill in a texture row.
      * @param texture_fill_j       The number of rows to fill in a texture image.
      * @param texture_fill_k       The number of images to fill into the texture.
      * @param pixels   A pointer to the contiguos memory location where texture
      *                 data should be loaded from.
      */
      template <typename T>
      void load_sub_image_into_texture(
        int image_skip_i
        , int image_skip_j
        , int image_skip_k
        , int image_size_i
        , int image_size_j
        , int image_size_k
        , int texture_offset_i
        , int texture_offset_j
        , int texture_offset_k
        , int texture_fill_i
        , int texture_fill_j
        , int texture_fill_k
        , T const * pixels
        )
      {
        assert(image_skip_i>=0 || !"Texture3D::load_sub_image_into_texture(): skip i was negative");
        assert(image_skip_j>=0 || !"Texture3D::load_sub_image_into_texture(): skip j was negative");
        assert(image_skip_k>=0 || !"Texture3D::load_sub_image_into_texture(): skip k was negative");

        assert(image_size_i>0 || !"Texture3D::load_sub_image_into_texture(): size i was non positive");
        assert(image_size_j>0 || !"Texture3D::load_sub_image_into_texture(): size j was non positive");
        assert(image_size_k>0 || !"Texture3D::load_sub_image_into_texture(): size k was non positive");

        assert(image_skip_i < image_size_i || !"Texture3D::load_sub_image_into_texture(): skip i was larger than size");
        assert(image_skip_j < image_size_j || !"Texture3D::load_sub_image_into_texture(): skip j was larger than size");
        assert(image_skip_k < image_size_k || !"Texture3D::load_sub_image_into_texture(): skip k was larger than size");

        assert(texture_offset_i>=0 || !"Texture3D::load_sub_image_into_texture(): offset i was negative");
        assert(texture_offset_j>=0 || !"Texture3D::load_sub_image_into_texture(): offset j was negative");
        assert(texture_offset_k>=0 || !"Texture3D::load_sub_image_into_texture(): offset k was negative");

        assert(texture_offset_i < m_texture_size_i || !"Texture3D::load_sub_image_into_texture(): offset i was larger than texture size");
        assert(texture_offset_j < m_texture_size_j || !"Texture3D::load_sub_image_into_texture(): offset j was larger than texture size");
        assert(texture_offset_k < m_texture_size_k || !"Texture3D::load_sub_image_into_texture(): offset k was larger than texture size");

        assert((texture_offset_i+texture_fill_i) <= m_texture_size_i || !"Texture3D::load_sub_image_into_texture(): texture fill exceeds texture size");
        assert((texture_offset_j+texture_fill_j) <= m_texture_size_j || !"Texture3D::load_sub_image_into_texture(): texture fill exceeds texture size");
        assert((texture_offset_k+texture_fill_k) <= m_texture_size_k || !"Texture3D::load_sub_image_into_texture(): texture fill exceeds texture size");

        assert(texture_fill_i <= (image_size_i - image_skip_i) || !"Texture3D::load_sub_image_into_texture(): texture fill i was larger than sub image region");
        assert(texture_fill_j <= (image_size_j - image_skip_j) || !"Texture3D::load_sub_image_into_texture(): texture fill j was larger than sub image region");
        assert(texture_fill_k <= (image_size_k - image_skip_k) || !"Texture3D::load_sub_image_into_texture(): texture fill k was larger than sub image region");

        //unsigned int ext_channels = 1;
        //m_format = external_format( ext_channels );
        //m_type   = external_type<T>();

        m_image_skip_i = image_skip_i;
        m_image_skip_j = image_skip_j;
        m_image_skip_k = image_skip_k;
        m_image_size_i = image_size_i;
        m_image_size_j = image_size_j;
        m_image_size_k = image_size_k;

        //std::cout << "Texture3D::load_sub_image_into_texture():"  << std::endl;
        //std::cout << "  skip i   = " << m_image_skip_i     << std::endl;
        //std::cout << "  skip j   = " << m_image_skip_j     << std::endl;
        //std::cout << "  skip k   = " << m_image_skip_k     << std::endl;
        //std::cout << "  size i   = " << m_image_size_i     << std::endl;
        //std::cout << "  size j   = " << m_image_size_j     << std::endl;
        //std::cout << "  size k   = " << m_image_size_k     << std::endl;
        //std::cout << "  offset i = " << texture_offset_i << std::endl;
        //std::cout << "  offset j = " << texture_offset_j << std::endl;
        //std::cout << "  offset k = " << texture_offset_k << std::endl;
        //std::cout << "  fill i   = " << texture_fill_i   << std::endl;
        //std::cout << "  fill j   = " << texture_fill_j   << std::endl;
        //std::cout << "  fill k   = " << texture_fill_k   << std::endl;

        assert(pixels!= 0 || !"texture3D::load_sub_image_into_texture() - pixels was null");
        bind();
        setup_pixel_transfer();

        glTexSubImage3D(
          m_texture_target         //GLenum target
          , 0                      //GLint level
          , texture_offset_i       //GLint x offset
          , texture_offset_j       //GLint y offset
          , texture_offset_k       //GLint z offset
          , texture_fill_i       //GLsizei width
          , texture_fill_j       //GLsizei height
          , texture_fill_k       //GLsizei depth
          , m_format               //GLenum format
          , m_type                 //GLenum type
          , pixels                 //const GLvoid *pixels
          );

        clear_pixel_transfer();
      }

      /**
      * Load Texture From Memory.
      *
      * @param pixels   A pointer to the contiguos memory location where texture
      *                 data should be loaded from.
      */
      template <typename T>
      void load(T const * pixels)
      {
        assert(pixels != 0 || !"Texture3D::load() - pixels are null");
        bind();
        setup_pixel_transfer();

        //unsigned int ext_channels = 1;
        //m_format = external_format( ext_channels );
        //m_type   = external_type<T>();

        int texture_offset_x = 0;
        int texture_offset_y = 0;
        int texture_offset_z = 0;
        glTexSubImage3D(
          m_texture_target         //GLenum target
          , 0                      //GLint level
          , texture_offset_x       //GLint x offset
          , texture_offset_y       //GLint y offset
          , texture_offset_z       //GLint z offset
          , m_texture_size_i       //GLsizei width
          , m_texture_size_j       //GLsizei height
          , m_texture_size_k       //GLsizei depth
          , m_format               //GLenum format
          , m_type                 //GLenum type
          , pixels                   //const GLvoid *pixels
          );

        clear_pixel_transfer();
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
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_R, GL_REPEAT);
        }
        else
        {
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
          glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
          //glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
          //glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP);
          //glTexParameteri( m_texture_target, GL_TEXTURE_WRAP_R, GL_CLAMP);
        }
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
        if (m_cubic)
        {
          if (sampler_type!= "sampler3DCube" && sampler_type!="samplerCUBE")
          {
            std::cerr << "Texture3D::validate_sampler_type(): TextureRectangle attempted to bound to sampler type " << sampler_type << std::endl;
            return false;
          }
        }
        else
        {
          if (sampler_type!="sampler3D")
          {
            std::cerr << "Texture3D::validate_sampler_type(): Attempted to bound to sampler type " << sampler_type << std::endl;
            return false;
          }
        }
        return true;
      }

    };

    typedef Texture3D*  texture3D_pointer;

  } // namespace texture
} // namespace OpenTissue

//OPENTISSUE_GPU_TEXTURE_TEXTURE_TEXTURE3D_H
#endif

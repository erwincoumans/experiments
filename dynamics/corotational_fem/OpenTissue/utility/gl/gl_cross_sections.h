#ifndef OPENTISSUE_UTILITY_GL_GL_CROSS_SECTIONS_H
#define OPENTISSUE_UTILITY_GL_GL_CROSS_SECTIONS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/gpu/image/image.h>
#include <cassert>
#include <cmath>

namespace OpenTissue
{
  namespace gl
  {

    /**
     * This color functor can be used to assign blue colors to positive
     * values and red to negative values.
     */
    class SignColorFunctor
    {
    public:
      template <typename value_type>
      void operator()(  value_type const & val, value_type const & /*min_val*/, value_type const & /*max_val*/, GLubyte & red, GLubyte & green, GLubyte & blue, GLubyte & alpha ) const
      {
        if(val>=0)
        {
          red   = 0;
          blue  = 255;
        }
        else
        {
          red   = 255;
          blue  = 0;
        }
        green = 0;
        alpha = 255;
      }
    };


    /**
     * This color functor can be used to make a grey scale ramp with black
     * corresponding to the minimum value and white to the maximum value.
     */
    class GreyScaleColorFunctor
    {
    public:
      template <typename value_type>
      void operator()( value_type const & val, value_type const & min_val, value_type const & max_val, GLubyte & red, GLubyte & green, GLubyte & blue, GLubyte & alpha ) const
      {
        value_type range = max_val - min_val;
        if(val>max_val || val<min_val)
        {
          red = green = blue = 0;
          alpha = 255;
          return;
        }
        value_type intensity = static_cast<value_type>( ( ( val - min_val ) * 255.0 ) / range );
        red = ( GLubyte ) ( intensity );
        green = ( GLubyte ) ( intensity );
        blue = ( GLubyte ) ( intensity );
        alpha = 255;
      }
    };

    /**
     * This color functor will create a red and blue ramp corresponding
     * to the negative and positive values.
     */
    class GradientColorFunctor
    {
    public:
      template <typename value_type>
      void operator()(  value_type const & val, value_type const & min_val, value_type const & max_val, GLubyte & red, GLubyte & green, GLubyte & blue, GLubyte & alpha ) const
      {
        value_type range = max_val - min_val;
        value_type red_intensity = static_cast<value_type>( ( ( max_val - val ) * 255.0 ) / range );
        value_type blue_intensity = static_cast<value_type>( ( ( val - min_val ) * 255.0 ) / range );
        red   = (GLubyte) ( red_intensity );
        green = 0;//(GLubyte) ( intensity );
        blue  = (GLubyte) ( blue_intensity );
        alpha = 255;
      }
    };


    /**
    * Cross Section Visualization Class.
    * This class contains functionality for extracting the intersections of
    * a 3D volume and three coordinate axes planes and visualizing the
    * resulting cross sections.
    */
    template <typename grid_type>
    class CrossSections
    {
    public:

      typedef typename grid_type::value_type                   value_type;
      typedef OpenTissue::image::Image<unsigned char>          image_type;

    protected:

      image_type  m_bitmap_i;    ///< A bitmap containing an image of the x-axis cross section.
      image_type  m_bitmap_j;    ///< A bitmap containing an image of the y-axis cross section.
      image_type  m_bitmap_k;    ///< A bitmap containing an image of the z-axis cross section.

      OpenTissue::texture::texture2D_pointer m_texture_i; ///< A texture corresponding to the x-axis image of the cross-section. This is the GPU stored image. Where as the bitmap are the CPU stored image.
      OpenTissue::texture::texture2D_pointer m_texture_j; ///< A texture corresponding to the y-axis image of the cross-section. This is the GPU stored image. Where as the bitmap are the CPU stored image.
      OpenTissue::texture::texture2D_pointer m_texture_k; ///< A texture corresponding to the z-axis image of the cross-section. This is the GPU stored image. Where as the bitmap are the CPU stored image.

      bool         m_changed_i; ///< Boolan flag indicating whether the x-axis plane have changed its position since the last draw-call.
      bool         m_changed_j; ///< Boolan flag indicating whether the y-axis plane have changed its position since the last draw-call.
      bool         m_changed_k; ///< Boolan flag indicating whether the z-axis plane have changed its position since the last draw-call.
      unsigned int m_i;         ///< A step counter along the x-axis, specifying the current location of that coordinate axis plane.
      unsigned int m_j;         ///< A step counter along the y-axis, specifying the current location of that coordinate axis plane.
      unsigned int m_k;         ///< A step counter along the z-axis, specifying the current location of that coordinate axis plane.
      unsigned int m_max_i;     ///< The maximum number of steps to be taken along the x-axis.
      unsigned int m_max_j;     ///< The maximum number of steps to be taken along the y-axis.
      unsigned int m_max_k;     ///< The maximum number of steps to be taken along the z-axis.
      grid_type  * m_ptr_grid;  ///< A pointer to the grid volume that should be visualized.

    public:

      CrossSections(  )
        : m_changed_i(false)
        , m_changed_j(false)
        , m_changed_k(false)
        , m_i(0)
        , m_j(0)
        , m_k(0)
        , m_max_i(0)
        , m_max_j(0)
        , m_max_k(0)
        , m_ptr_grid(0)
      {}

    protected:

      /**
      * Get Axis Bitmap and Update Texture.
      * This method retrives the cross intersection with the x-axis plane. The
      * cross intersection is stored in the corresponding axis bitmap and then
      * the corresponding texture is updated with the content of the bitmap.
      * 
      * @param color_func    A color function funtor. This functor is used
      *                      to translate the grid values from the cross section
      *                      into color values (RGBA values stored as a (255,255,255,255)
      *                      color format).
      */
      template<typename color_func_type>
      void get_i_cut(color_func_type & color_func  )
      {
        using OpenTissue::grid::min_element;
        using OpenTissue::grid::max_element;

        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;

        GLubyte * imgdata = static_cast<GLubyte*>( m_bitmap_i.get_data() );

        assert( m_i >= 0u && m_i < m_ptr_grid->I() );

        value_type min_val = min_element( *m_ptr_grid );
        value_type max_val = max_element( *m_ptr_grid );
        for ( unsigned int k = 0;k < m_ptr_grid->K();++k )
        {
          for ( unsigned int j = 0;j < m_ptr_grid->J();++j )
          {
            GLubyte * red = &imgdata[ ( k * m_ptr_grid->J() + j ) * 4 ];
            GLubyte * green = red + 1;
            GLubyte * blue = green + 1;
            GLubyte * alpha = blue + 1;
            unsigned int idx = ( k * m_ptr_grid->I() * m_ptr_grid->J() ) + ( j * m_ptr_grid->I() ) + m_i;
            value_type val = (*m_ptr_grid)(idx);
            color_func( val, min_val, max_val, *red, *green, *blue, *alpha );
          }
        }

        m_texture_i->load( m_bitmap_i.get_data() );
      }

      /**
      * Get Axis Bitmap and Update Texture.
      * This method retrives the cross intersection with the x-axis plane. The
      * cross intersection is stored in the corresponding axis bitmap and then
      * the corresponding texture is updated with the content of the bitmap.
      * 
      * @param color_func    A color function funtor. This functor is used
      *                      to translate the grid values from the cross section
      *                      into color values (RGBA values stored as a (255,255,255,255)
      *                      color format).
      */
      template<typename color_func_type>
      void get_j_cut(color_func_type & color_func )
      {
        using OpenTissue::grid::min_element;
        using OpenTissue::grid::max_element;

        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;

        GLubyte * imgdata = static_cast<GLubyte*>( m_bitmap_j.get_data() );
        assert( m_j >= 0u && m_j < m_ptr_grid->J() );
        value_type min_val = min_element( *m_ptr_grid );
        value_type max_val = max_element( *m_ptr_grid );
        for ( unsigned int k = 0; k < m_ptr_grid->K(); ++k )
        {
          for ( unsigned int i = 0; i < m_ptr_grid->I(); ++i )
          {
            GLubyte * red = &imgdata[ ( k * m_ptr_grid->I() + i ) * 4 ];
            GLubyte * green = red + 1;
            GLubyte * blue = green + 1;
            GLubyte * alpha = blue + 1;
            unsigned int idx = ( k * m_ptr_grid->I() * m_ptr_grid->J() ) + ( m_j * m_ptr_grid->I() ) + i;
            value_type val = (*m_ptr_grid)(idx);
            color_func( val, min_val, max_val, *red, *green, *blue, *alpha );
          }
        }

        m_texture_j->load( m_bitmap_j.get_data() );
      }

      /**
      * Get Axis Bitmap and Update Texture.
      * This method retrives the cross intersection with the x-axis plane. The
      * cross intersection is stored in the corresponding axis bitmap and then
      * the corresponding texture is updated with the content of the bitmap.
      * 
      * @param color_func    A color function funtor. This functor is used
      *                      to translate the grid values from the cross section
      *                      into color values (RGBA values stored as a (255,255,255,255)
      *                      color format).
      */
      template<typename color_func_type>
      void get_k_cut( color_func_type & color_func )
      {
        using OpenTissue::grid::min_element;
        using OpenTissue::grid::max_element;

        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;

        GLubyte * imgdata = static_cast<GLubyte*>( m_bitmap_k.get_data() );
        assert( m_k >= 0u && m_k < m_ptr_grid->K() );
        value_type min_val = min_element( *m_ptr_grid );
        value_type max_val = max_element( *m_ptr_grid );
        for ( unsigned int j = 0; j < m_ptr_grid->J(); ++j )
        {
          for ( unsigned int i = 0; i < m_ptr_grid->I(); ++i )
          {
            GLubyte * red = &imgdata[ ( j * m_ptr_grid->I() + i ) * 4 ];
            GLubyte * green = red + 1;
            GLubyte * blue = green + 1;
            GLubyte * alpha = blue + 1;
            unsigned int idx = ( m_k * m_ptr_grid->I() * m_ptr_grid->J() ) + ( j * m_ptr_grid->I() ) + i;
            value_type val = (*m_ptr_grid)(idx);
            color_func( val, min_val, max_val, *red, *green, *blue, *alpha );
          }
        }

        m_texture_k->load( m_bitmap_k.get_data() );
      }

      /**
      * Get Axes Bitmaps and Update Textures.
      * This method determines which coordinate axes planes that have
      * changed. Only for the changed axes are bitmaps retrieved and textures updated.
      * 
      * @param color_func    A color function funtor. This functor is used
      *                      to translate the grid values from the cross section
      *                      into color values (RGBA values stored as a (255,255,255,255)
      *                      color format).
      */
      template<typename color_func_type>
      void get_bitmaps(  color_func_type & color_func  )
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;

        if(m_max_i != m_ptr_grid->I() || m_max_j != m_ptr_grid->J() || m_max_k != m_ptr_grid->K() )
          allocate();
        if(m_changed_i)
        {
          get_i_cut ( color_func );
          m_changed_i = false;
        }
        if(m_changed_j)
        {
          get_j_cut ( color_func );
          m_changed_j = false;
        }
        if(m_changed_k)
        {
          get_k_cut ( color_func );
          m_changed_k = false;
        }
      }

      void allocate()
      {
        m_max_i = m_ptr_grid->I();
        m_max_j = m_ptr_grid->J();
        m_max_k = m_ptr_grid->K();
        m_bitmap_i.create(m_max_j,m_max_k,4);
        m_bitmap_j.create(m_max_i,m_max_k,4);
        m_bitmap_k.create(m_max_i,m_max_j,4);
        m_texture_i = m_bitmap_i.create_texture(GL_RGBA, m_max_j!=m_max_k );
        m_texture_j = m_bitmap_j.create_texture(GL_RGBA, m_max_i!=m_max_k );
        m_texture_k = m_bitmap_k.create_texture(GL_RGBA, m_max_i!=m_max_j );
        m_i = m_max_i/2;
        m_j = m_max_j/2;
        m_k = m_max_k/2;
        m_changed_i =  m_changed_j = m_changed_k = true;
      }

    public:

      /**
      * Move x-plane.
      * This method should be invoked to move the x-plane one step
      * along the positive direction of the x-axis. One can not
      * move beyond the maximum step limit.
      */
      void forward_i_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_i = (m_i + 1)%m_max_i;
        m_changed_i = true;
      }

      /**
      * Move x-plane.
      * This method should be invoked to move the x-plane one step
      * along the negative direction of the x-axis. One can not
      * move beyond the zero step limit.
      */
      void backward_i_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_i = (m_i - 1)%m_max_i;
        if(m_i<0)
          m_i += m_max_i;
        m_changed_i = true;
      }

      /**
      * Move y-plane.
      * This method should be invoked to move the y-plane one step
      * along the positive direction of the y-axis. One can not
      * move beyond the maximum step limit.
      */
      void forward_j_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_j = (m_j + 1)%m_max_j;
        m_changed_j = true;
      }

      /**
      * Move y-plane.
      * This method should be invoked to move the y-plane one step
      * along the negative direction of the y-axis. One can not
      * move beyond the zero step limit.
      */
      void backward_j_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_j = (m_j - 1)%m_max_j;
        if(m_j<0)
          m_j += m_max_j;
        m_changed_j = true;
      }

      /**
      * Move z-plane.
      * This method should be invoked to move the z-plane one step
      * along the positive direction of the z-axis. One can not
      * move beyond the maximum step limit.
      */
      void forward_k_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_k = (m_k + 1)%m_max_k;
        m_changed_k = true;
      }

      /**
      * Move z-plane.
      * This method should be invoked to move the z-plane one step
      * along the negative direction of the z-axis. One can not
      * move beyond the zero step limit.
      */
      void backward_k_plane()
      {
        if(!m_ptr_grid)
          return;
        if(!(m_ptr_grid->size()))
          return;
        m_k = (m_k - 1)%m_max_k;
        if(m_k<0)
          m_k += m_max_k;
        m_changed_k = true;
      }

    public:

      /**
      * Invalidate Visualization.
      * Invoking this method will force the visualization to re-fetch
      * all cross intersections and store them in textures. The following
      * draw-invokation will then be a complete re-draw of the data.
      *
      * It is generally a good idea to invoke this method when data stored
      * in the grid volume is altered.
      */
      void invalidate()
      {
        if(!m_ptr_grid)
          return;

        if(!(m_ptr_grid->size()))
          return;

        m_changed_i =  m_changed_j = m_changed_k = true;
      }

      /**
      * Initialization.
      * This method should be invoked before any visualization takes
      * place. The method will allocate and initialize any internal
      * data structures that is needed for the cross section visualization.
      *
      * Observe that this class is completely bound to one grid. If
      * one has multiple grids that need to be visualized. Then a
      * seperate instance of this class is needed for each visualization
      * wanted.
      *
      * @param grid     The grid that should be visualized.
      */
      void init(grid_type const & grid)
      {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        m_ptr_grid = const_cast<grid_type*>(&grid);
        allocate();
        glPopAttrib();
      }

      /**
      * Draw Cross Sections.
      * Invoke this method in ones render-loop.
      *
      * @param color_func    A color function funtor. This functor is used
      *                      to translate the grid values from the cross sections
      *                      into color values (RGBA values stored as a (255,255,255,255)
      *                      color format).
      */
      template<typename color_func_type>
      void draw(  color_func_type const & color_func  )
      {
        gl::gl_check_errors("OpenTissue::CrossSections::draw() - start");

        if(!m_ptr_grid)
          return;

        if(!(m_ptr_grid->size()))
          return;

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glDisable(GL_LIGHTING);

        get_bitmaps(color_func);

        GLfloat sx = ( m_ptr_grid->max_coord(0) - m_ptr_grid->min_coord(0) ) / m_ptr_grid->I();
        GLfloat sy = ( m_ptr_grid->max_coord(1) - m_ptr_grid->min_coord(1) ) / m_ptr_grid->J();
        GLfloat sz = ( m_ptr_grid->max_coord(2) - m_ptr_grid->min_coord(2) ) / m_ptr_grid->K();

        GLfloat tx = m_i * m_ptr_grid->dx();
        GLfloat ty = m_j * m_ptr_grid->dy();
        GLfloat tz = m_k * m_ptr_grid->dz();

        GLfloat cx = m_ptr_grid->min_coord(0);
        GLfloat cy = m_ptr_grid->min_coord(1);
        GLfloat cz = m_ptr_grid->min_coord(2);

        GLfloat Ival = 1.0f*m_ptr_grid->I();
        GLfloat Jval = 1.0f*m_ptr_grid->J();
        GLfloat Kval = 1.0f*m_ptr_grid->K();

        glPushMatrix();
        glTranslatef( cx, cy, cz );
        glPushMatrix();
        glTranslatef( tx, 0, 0 );
        glRotatef( 90, 0, 0, 1 );
        glRotatef( 90, 1, 0, 0 );
        glScalef( sy, sz, 1 );
        gl::ColorPicker( 1.0, 0, 0, 1.0 );
        glBegin( GL_LINE_LOOP ); //--- Draw Frame
        glVertex3f( 0, 0, 0 );
        glVertex3f( Jval, 0, 0 );
        glVertex3f( Jval, Kval, 0 );
        glVertex3f( 0, Kval, 0 );
        glEnd();
        gl::ColorPicker( 1.0, 1.0, 1.0, 1.0 );
        gl::DrawTexture2D(*m_texture_i);
        glPopMatrix();

        glPushMatrix();
        glTranslatef( 0, ty, 0 );
        glRotatef( 90, 1, 0, 0 );
        glScalef( sx, sz, 1 );
        gl::ColorPicker( 0, 1.0, 0, 1.0 );
        glBegin( GL_LINE_LOOP ); //--- Draw Frame
        glVertex3f( 0, 0, 0 );
        glVertex3f( Ival, 0, 0 );
        glVertex3f( Ival, Kval, 0 );
        glVertex3f( 0, Kval, 0 );
        glEnd();
        gl::ColorPicker( 1.0, 1.0, 1.0, 1.0 );
        gl::DrawTexture2D(*m_texture_j);
        glPopMatrix();

        glPushMatrix();
        glTranslatef( 0, 0, tz );
        glScalef( sx, sy, 1 );
        gl::ColorPicker( 0, 0, 1.0, 1.0 );
        glBegin( GL_LINE_LOOP ); //--- Draw Frame
        glVertex3f( 0, 0, 0 );
        glVertex3f( Ival, 0, 0 );
        glVertex3f( Ival, Jval, 0 );
        glVertex3f( 0, Jval, 0 );
        glEnd();
        gl::ColorPicker( 1.0, 1.0, 1.0, 1.0 );
        gl::DrawTexture2D(*m_texture_k);
        glPopMatrix();

        glPopMatrix();
        glPopAttrib();
      }

    };

  } // namespace gl
} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_GL_CROSS_SECTIONS_H
#endif

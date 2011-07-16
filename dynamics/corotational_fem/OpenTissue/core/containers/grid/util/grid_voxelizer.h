#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_VOXELIZER_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_VOXELIZER_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl_util.h>
#include <OpenTissue/gpu/texture/texture_create_texture2D.h>
#include <OpenTissue/core/math/math_vector3.h>
#include <OpenTissue/core/containers/mesh/mesh.h>
#include <OpenTissue/core/containers/grid/grid.h>

#include <boost/cast.hpp> //--- Needed for boost::numerical_cast

#include <cmath>
#include <iostream>

namespace OpenTissue
{
  namespace grid
  {

    namespace detail
    {

      template<typename grid_type>
      class Voxelizer
      {
      public:

        typedef typename grid_type::value_type            T;
        typedef typename grid_type::math_types            math_types;
        typedef typename math_types::real_type           real_type;
        typedef typename math_types::vector3_type        vector3_type;

      protected:

        std::vector<GLubyte>      m_pixels;  ///< Big data chunck used to examine pixels in the frame (or pixel) buffer

        OpenTissue::texture::texture2D_pointer   m_texture;
        gl::renderbuffer_pointer                 m_stencil;
        gl::renderbuffer_pointer                 m_color;
        gl::FramebufferObject                    m_fbo;

      public:

        Voxelizer()
          : m_pixels()
        {}

        ~Voxelizer(){}

      public:

        template<typename mesh_type>
        void run(mesh_type & mesh, grid_type & voxels)
        {
          using std::min;
          using std::max;

          if(mesh.size_faces()==0)
            return;

          //--- Initialize
          std::fill(voxels.begin(),voxels.end(), boost::numeric_cast<T>(0) );
          m_pixels.resize( voxels.I()*voxels.J()*4 );

          glPushAttrib(GL_ALL_ATTRIB_BITS);
          gl::gl_check_errors("voxelizer(): push all atrribs");

          int w = voxels.I(); 
          int h = voxels.J();

          bool fbo_support = true;
          if(fbo_support)
          {
            m_stencil = gl::create_packed_depth_and_stencil_buffer(w,h);
            gl::gl_check_errors("voxelizer(): stencil buffer created");
            m_color.reset( new gl::Renderbuffer(GL_RGBA8_EXT, w, h) );
            gl::gl_check_errors("voxelizer(): color buffer created");
            m_fbo.bind();
            gl::gl_check_errors("voxelizer(): fbo bind");
            m_fbo.attach_render_buffer(GL_STENCIL_ATTACHMENT_EXT, m_stencil);
            gl::gl_check_errors("voxelizer(): attach stencil buffer");
            m_fbo.attach_render_buffer(GL_DEPTH_ATTACHMENT_EXT, m_stencil);//--- experiment with packed depth and stencil buffer
            gl::gl_check_errors("voxelizer(): attach stencil buffer");      
            m_fbo.attach_render_buffer(GL_COLOR_ATTACHMENT0_EXT, m_color);
            gl::gl_check_errors("voxelizer(): attach color buffer");
            m_fbo.is_valid();
            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
          }

          glDisable(GL_COLOR_MATERIAL);
          glDisable(GL_LIGHTING);
          glMatrixMode( GL_PROJECTION );
          glPushMatrix();
          glLoadIdentity();
          GLdouble left   =  voxels.min_coord(0);
          GLdouble right  =  voxels.max_coord(0);
          GLdouble bottom =  voxels.min_coord(1);
          GLdouble top    =  voxels.max_coord(1);
          left   -= voxels.dx()/2;
          right  += voxels.dx()/2;
          bottom -= voxels.dy()/2;
          top    += voxels.dy()/2;
          GLdouble zdiff = 2.*(voxels.max_coord(2) - voxels.min_coord(2));
          glOrtho(left,right,bottom,top,-zdiff,zdiff);
          glViewport(0,0,w,h);
          glMatrixMode( GL_MODELVIEW );
          glPushMatrix();
          glLoadIdentity();

          //--- scan z-planes
          vector3_type vmin,vmax;
          mesh::compute_mesh_minimum_coord(mesh,vmin);
          mesh::compute_mesh_maximum_coord(mesh,vmax);

          //--- Now we are ready for clipping and scanconverting
          int kmin = boost::numeric_cast<int>((vmin(2) - voxels.min_coord(2))/voxels.dz()) - 1;
          kmin = max(kmin,0);
          int kmax = boost::numeric_cast<int>((vmax(2) - voxels.min_coord(2))/voxels.dz()) + 1;
          kmax = min(kmax,boost::numeric_cast<int>(voxels.K()));

          for(int k=kmin;k<kmax;++k)
            clip(&mesh, voxels, k, vmin, vmax);

          //--- cleanup
          if(fbo_support)
            gl::FramebufferObject::disable();

          glMatrixMode( GL_MODELVIEW );
          glPopMatrix();
          glMatrixMode( GL_PROJECTION );
          glPopMatrix();
          glPopAttrib();
        }

      private:

        void draw_rectangle(grid_type & voxels, real_type const & minx,real_type const & miny,real_type const & maxx,real_type const & maxy)
        {
          glBegin(GL_POLYGON);
          glVertex3d(minx,miny,voxels.max_coord(2));
          glVertex3d(maxx,miny,voxels.max_coord(2));
          glVertex3d(maxx,maxy,voxels.max_coord(2));
          glVertex3d(minx,maxy,voxels.max_coord(2));
          glEnd();
        }

        /**
        * Clip-Intersection Routine.
        * Clip object with z-cutting plane given by the k-index along
        * the z-axe and draw intersection (pixels inside object and
        * lying on cutting plane).
        *
        * @param k       Index of z-slice that should be intersected against object.
        */
        template<typename mesh_type>
        void clip(mesh_type * mesh, grid_type & voxels, int k,vector3_type & vmin,vector3_type & vmax)
        {
          //--- Eye is far out the positive z-axe, rays are shot into
          //--- the eye, the clip plane clips the part of the object
          //--- already sweeped (ie. in the negative direction of the z-axe).

          //--- Compute Cutting Plane
          {
            GLdouble zcut = (k * voxels.dz()) + voxels.min_coord(2);

            GLdouble planeEq[4];
            planeEq[0] = 0;
            planeEq[1] = 0;
            planeEq[2] = 1;
            planeEq[3] = -zcut;

            glClipPlane(GL_CLIP_PLANE0,planeEq);
            glEnable(GL_CLIP_PLANE0);
          }

          //--- Clear everything
          {
            glClearColor( 0.0, 0.0, 0.0, 0.0 );
            glClearDepth(0);
            glDepthMask(GL_TRUE);
            glClearStencil(0);
            glStencilMask(~0u);
            glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
          }

          //--- First Pass
          {
            glEnable(GL_STENCIL_TEST);                        //--- Turn on stencil testing
            glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE); //--- Do not draw to pixel buffer
            glDisable(GL_DEPTH_TEST);                         //--- Disable z-buffer
            glEnable(GL_CULL_FACE);
            glStencilFunc(GL_ALWAYS,0,~0u);
            glCullFace(GL_BACK);                              //--- Draw only front facing faces
            glStencilOp(GL_KEEP,GL_KEEP,GL_INCR);             //--- Every time a pixel of a front facing face
            //--- is rendered increment stencil buffer by one
            glColor3f(1.0,0.0,0.0);
            gl::DrawMesh(*mesh,GL_POLYGON,false,false,false); //--- Draw object (first pass)
          }

          //--- Second Pass
          {
            glCullFace(GL_FRONT);                             //--- Draw only back facing faces
            glStencilOp(GL_KEEP,GL_KEEP,GL_DECR);             //--- Every time a pixel of a back facing face
            //--- is rendered decrement stencil buffer by one
            glColor3f(0.0,1.0,0.0);
            gl::DrawMesh(*mesh,GL_POLYGON,false,false,false); //--- Draw object (second pass)
          }

          //--- Third Pass
          {
            glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);   //--- Turn on drawing to the pixel buffer
            glStencilFunc(GL_EQUAL,1,~0u);                  //--- Only draw where stencil buffer is one
            glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);           //--- Do not alter stencil buffer

            glCullFace(GL_BACK);
            glColor3f(0,0,1);
            draw_rectangle(voxels,vmin(0),vmin(1),vmax(0),vmax(1));    //--- Draw pixels of intersection region (Third pass)
          }

          extract_voxels(voxels,k);

          //--- Re-establish normal drawing mode
          {
            glDisable(GL_CLIP_PLANE0);
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_STENCIL_TEST);
            glCullFace(GL_BACK);
          }
        }

        template<typename grid_type2>
        void extract_voxels(grid_type2 & voxels, int k)
        {
          gl::gl_check_errors("extract_voxels::clip(): on entry");

          int w = boost::numeric_cast<int>( voxels.I() );
          int h = boost::numeric_cast<int>( voxels.J() );

          glFinish();
          gl::gl_check_errors("voxelizer::clip(): glFinish");
          glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, &m_pixels[0] );
          gl::gl_check_errors("extract_voxels::clip(): glReadPixels");
          for( int j=0;j<h;++j)
          {
            for( int i=0;i<w;++i)
            {
              int offset = (w*j + i)*4;
              //unsigned char red   = m_pixels[ offset    ];
              //unsigned char green = m_pixels[ offset + 1];
              unsigned char blue  = m_pixels[ offset + 2];
              //unsigned char alpha = m_pixels[ offset + 3];
              if(blue>0)
                voxels(i,j,k) = 1;
              else
                voxels(i,j,k) = 0;
            }
          }
        }

      };

    } // namespace detail

    /**
    * Voxelizer.
    *
    * @param mesh   The mesh that should be converted into a voxel grid.
    * @param phi    Upon return this grid holds the voxels (1: voxel, 0: no voxel).
    */
    template<typename mesh_type,typename grid_type>
    inline void voxelizer(mesh_type & mesh, grid_type & phi)
    {
      typedef detail::Voxelizer<grid_type> voxelizer_type;
      voxelizer_type().run(mesh,phi);
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_VOXELIZER_H
#endif

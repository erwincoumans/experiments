#ifndef OPENTISSUE_UTILITY_GL_GL_DRAW_MESH_H
#define OPENTISSUE_UTILITY_GL_GL_DRAW_MESH_H
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
    * This Drawing Utility is based on Vertex Arrays.
    *
    * Since a mesh may not be an array, this utility requires some
    * setting up, but once it have been initialized a mesh can be drawn
    * quickly and efficietly.
    *
    * NOTE: The utility only supports triangular meshes.
    */
    template<typename mesh_type>
    class MeshDrawArray
    {
    protected:

      typedef typename mesh_type::vertex_traits             vertex_data;
      typedef typename mesh_type::math_types::vector3_type  vector3_type;
      typedef typename mesh_type::math_types::real_type     real_type;

      vertex_data  * m_data;      ///< vertex data: (coord, normal, u,v, color3)
      unsigned int * m_indices;   ///< vertex indices of triangles
      std::size_t    m_count;     ///< Number of triangles

    public:

      MeshDrawArray()
        : m_data(0)
        , m_indices(0)
        , m_count(0)
      { }

      MeshDrawArray(mesh_type const & mesh)
      {
        //--- We can not simply use the vertex iterators as points, because
        //--- we have no idea of what kind of kernel that is being used. Vertex
        //--- data may therefore not be nicely packed in memory!!!
        {
          m_data = new vertex_data[mesh.size_vertices()];
          typename mesh_type::const_vertex_iterator end  = mesh.vertex_end();
          typename mesh_type::const_vertex_iterator v    = mesh.vertex_begin();
          vertex_data * data = m_data;
          for(;v!=end;++v,++data)
          {
            //(*data) = *( static_cast<vertex_data>(v));
            (*data).m_coord = v->m_coord;
            (*data).m_normal = v->m_normal;
            (*data).m_u = v->m_u;
            (*data).m_v = v->m_v;
            (*data).m_color = v->m_color;
            //std::cout << "mc: " << (*data).m_coord << std::endl;
          }
        }

        //std::cout << "Vertices size: " << mesh.size_vertices() << std::endl;

        //--- Faces do not store vertex indices explicity in the polymesh
        //--- data structure, so we must create an index array outselves.
        {
          m_count  = 3*mesh.size_faces();
          m_indices = new unsigned int[m_count];
          typename mesh_type::const_face_iterator end = mesh.face_end();
          typename mesh_type::const_face_iterator f   = mesh.face_begin();
          unsigned int * indices = m_indices;
          for(;f!=end;++f)
          {
            assert(valency(*f)==3 || !"Only triangles are supported");
            typename mesh_type::const_face_vertex_circulator v( *f );

            *indices = static_cast<unsigned int>( v->get_handle().get_idx() );
            ++indices;
            ++v;

            *indices = static_cast<unsigned int>( v->get_handle().get_idx() );
            ++indices;
            ++v;

            *indices = static_cast<unsigned int>( v->get_handle().get_idx() );
            ++indices;
            ++v;
          }
        }
      }

      ~MeshDrawArray()
      {
        if(m_data)
          delete [] m_data;
        if(m_indices)
          delete [] m_indices;
      };

    public:

      void operator()()const
      {
        if(m_count==0)  return;

        assert(m_data || !"No data to show?");
        assert(m_indices || !"No data to show?");
        assert(m_count || !"No data to show?");

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY );

        GLvoid * ptr_coord     = m_data;
        GLvoid * ptr_normal    = &(m_data->m_normal);
        GLvoid * ptr_tex       = &(m_data->m_u);
        GLvoid * ptr_color     = &(m_data->m_color);

        GLsizei stride         = sizeof(vertex_data);
        //std::cout << "Stride: " << stride << std::endl;
        unsigned int real_size = sizeof(real_type);
        switch(real_size)
        {
        case 4:
          glVertexPointer  ( 3, GL_FLOAT, stride, ptr_coord  );
          glNormalPointer  (    GL_FLOAT, stride, ptr_normal );
          glTexCoordPointer( 2, GL_FLOAT, stride, ptr_tex    );
          glColorPointer   ( 3, GL_FLOAT, stride, ptr_color  );
          break;
        case 8:
          glVertexPointer  ( 3, GL_DOUBLE, stride, ptr_coord  );
          glNormalPointer  (    GL_DOUBLE, stride, ptr_normal );
          glTexCoordPointer( 2, GL_DOUBLE, stride, ptr_tex    );
          glColorPointer   ( 3, GL_DOUBLE, stride, ptr_color  );
          break;
        default:
          assert(!"Could not deduce data type");
          break;
        };
        //std::cout << "Drawing..." << std::endl;
        glDrawElements(GL_TRIANGLES, (GLsizei)m_count, GL_UNSIGNED_INT, m_indices);

        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY );
      }
    };


    /**
    * Simple brute force openGL drawing.
    * This is usefull for debugging, since it is extremely
    * robust. However, it is also extremely slow.
    *
    * @param face       The mesh face that should be drawn.
    * @param wireframe  whether the face should be drawn in wireframe or normal (solid).
    */
    template<typename face_type>
    inline void DrawMeshFace(face_type const & face, bool wireframe = false, bool use_colors = true, bool use_normals = true, bool use_texcoords = true)
    {
      glBegin(wireframe ? GL_LINE_LOOP : GL_POLYGON);
      typename face_type::mesh_type::const_face_vertex_circulator v(face), vend;
      for(; v != vend; ++v)
      {
        if(use_normals)
        {
          glNormal3f( (GLfloat) v->m_normal(0), (GLfloat)v->m_normal(1), (GLfloat)v->m_normal(2) );
        }
        if(use_texcoords)
        {
          glTexCoord2f( (GLfloat) v->m_u, (GLfloat)v->m_v );
        }
        if(use_colors)
        {
          glColor3f( (GLfloat) v->m_color(0), (GLfloat)v->m_color(1), (GLfloat)v->m_color(2) );
        }
        glVertex3f  ( (GLfloat) v->m_coord(0), (GLfloat)v->m_coord(1), (GLfloat)v->m_coord(2) );
      }
      glEnd();
    }


    /**
    * Simple brute force openGL drawing.
    * This is usefull for debugging, since it is extremely
    * robust. However, it is also extremely slow.
    *
    * @param mesh     The mesh that should be drawn.
    * @param mode     The mode that the polygon mesh should be drawn in (GL_POLYGON or GL_LINE_LOOP are accepted).
    */
    template<typename mesh_type>
    inline void DrawMesh(mesh_type const & mesh, unsigned int mode = GL_POLYGON, bool use_colors = true, bool use_normals = true, bool use_texcoords = true)
    {
      assert(mode==GL_POLYGON || mode==GL_LINE_LOOP || !"Illegal opengl mode");
      typename mesh_type::const_face_iterator fend = mesh.face_end();
      typename mesh_type::const_face_iterator f    = mesh.face_begin();

      for(;f!=fend;++f)
      {
        DrawMeshFace(*f, mode == GL_LINE_LOOP, use_colors, use_normals, use_texcoords);
      }
    }


    template<typename mesh_type>
    inline void DrawSkinMesh(mesh_type const & mesh, unsigned int mode = GL_POLYGON  )
    {
      assert(mode==GL_POLYGON || mode==GL_LINE_LOOP || !"Illegal opengl mode");

      typedef typename mesh_type::math_types::real_type			real_type;

      typename mesh_type::const_face_iterator fend = mesh.face_end();
      typename mesh_type::const_face_iterator f    = mesh.face_begin();

      for(;f!=fend;++f)
      {
        typename mesh_type::const_face_vertex_circulator v(*f),vend;
        glBegin(mode);
        //int maxi = 0;
        for(;v!=vend;++v)
        {
          //glTexCoord2f( (GLfloat) v->m_u, (GLfloat)v->m_v );
          //glColor3f( (GLfloat) v->m_color(0), (GLfloat)v->m_color(1), (GLfloat)v->m_color(2) );


          real_type weights[] = { 0, 0, 0, 0 };
          size_t bones[] = { 0, 0, 0, 0 };

          // TODO: check that m_influences is always < 4
          assert( v->m_influences < 5 || !"Only up to 4 bone influces is supported by GPU skinning!" );
          for(int i= 0; i < v->m_influences; ++i)
          {
            weights[i] = v->m_weight[i];
            bones[i] = v->m_bone[i];
            //maxi = std::max( maxi, bones[i] );
            //std::cout << bones[i] << " ";
          }
          //std::cout << std::endl;

          // Weights
          glColor4f( weights[0], weights[1], weights[2], weights[3] );

          // Bone indices
          glTexCoord4f( bones[0], bones[1], bones[2], bones[3] );

          /*
          typename skeleton_type::bone_type const * bone = skeleton.get_bone(bones[0]);
          typename skeleton_type::vector3_type vec = v->m_coord;
          bone->bone_space_transform().xform_point( vec );
          glVertex3f  ( (GLfloat) vec(0), (GLfloat)vec(1), (GLfloat)vec(2) );
          //*/

          glNormal3f( (GLfloat) v->m_normal(0), (GLfloat)v->m_normal(1), (GLfloat)v->m_normal(2) );
          glVertex3f  ( (GLfloat) v->m_coord(0), (GLfloat)v->m_coord(1), (GLfloat)v->m_coord(2) );


        }
        //std::cout << "Max_i: " << maxi << std::endl;
        glEnd();
      }
    }

    /**
    * Display List based Rendring.
    * This is a little slower than using vertex array, but it allows
    * for arbitary polygons.
    *
    * There is a certain maximum size on a display list. This class therefore
    * chops up a polygon mesh into several display lists. The size of these
    * is currently hardwired to 50.000 polygons.
    *
    */
    template<typename mesh_type>
    class MeshDrawDisplayLists
    {
    protected:

      GLuint m_lists;   ///< The identifier of the first display list.
      GLuint m_range;   ///< The number of display lists used.

    private:

      void create_display_list(mesh_type const & mesh, unsigned int mode = GL_POLYGON, bool use_colors = true, bool use_normals = true, bool use_texcoords = true)
      {
        using std::ceil;
        assert(mode==GL_POLYGON || mode==GL_LINE_LOOP || !"Illegal opengl mode");

        unsigned int total_count  = static_cast<unsigned int>( mesh.size_faces() );            //--- total number of polygons in mesh
        if(total_count == 0)
        {
          m_range = m_lists = 0u;
          return;
        }
        unsigned int list_count = std::min( total_count, 50000u );                              //--- maximum number of polygons allowed in a display list
        m_range = static_cast<int>( ceil( total_count / static_cast<float>( list_count ) ) );  //--- the number of display lists
        unsigned int count_left = total_count;                                                 //--- number of polygons that still need to be added to display lists
        m_lists = glGenLists( m_range );                                                       //--- generate m_range empty display lists

        typename mesh_type::const_face_iterator f = mesh.face_begin();
        for ( GLuint number = 0; number < m_range; ++number )
        {
          glNewList( m_lists + number, GL_COMPILE );
          unsigned int count = std::min(count_left,list_count);
          for ( unsigned int i = 0; i < count; ++i, ++f )
          {
            typename mesh_type::const_face_vertex_circulator v(*f),vend;
            glBegin(mode);
            for(;v!=vend;++v)
            {
              if(use_normals)
                glNormal3f  ( (GLfloat) v->m_normal(0), (GLfloat)v->m_normal(1), (GLfloat)v->m_normal(2) );
              if(use_texcoords)
                glTexCoord2f( (GLfloat) v->m_u        , (GLfloat)v->m_v                                  );
              if(use_colors)
                glColor3f   ( (GLfloat) v->m_color(0) , (GLfloat)v->m_color(1) , (GLfloat)v->m_color(2)  );
              glVertex3f  ( (GLfloat) v->m_coord(0) , (GLfloat) v->m_coord(1) , (GLfloat) v->m_coord(2)  );
            }
            glEnd();
          }
          count_left -= count;
          glEndList();
        }
      };

      void delete_display_lists()
      {
        if ( m_lists )
          glDeleteLists( m_lists, m_range );
        m_lists = 0;
        m_range = 0;
      };

      void call_display_lists()
      {
        for ( GLuint number = 0; number < m_range; ++number )
          glCallList( m_lists + number );
      };

    public:

      MeshDrawDisplayLists()
        :  m_lists(0)
        , m_range (0)
      {};

      MeshDrawDisplayLists(mesh_type const & mesh, unsigned int mode = GL_POLYGON, bool use_colors = true, bool use_normals = true, bool use_texcoords = true)
      {
        create_display_list(mesh,mode,use_colors,use_normals,use_texcoords);
      };

      ~MeshDrawDisplayLists(){delete_display_lists();};

    public:

      void operator()(){  call_display_lists(); };
    };

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_GL_DRAW_MESH_H
#endif

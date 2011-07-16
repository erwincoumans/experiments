
// OpenTissue Template Library Demo
// - A specific demonstration of the flexibility of OTTL.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL and OTTL Demos are licensed under zlib.
//

#include <OpenTissue/configuration.h>

#define DEFINE_GLUT_MAIN
#include <OpenTissue/utility/glut/glut_perspective_view_application.h>
#undef DEFINE_GLUT_MAIN


#include <OpenTissue/core/math/math_basic_types.h>
#include <OpenTissue/dynamics/fem/fem.h>

#include <OpenTissue/core/containers/t4mesh/util/t4mesh_block_generator.h>

//#include <OpenTissue/utility/utility_timer.h>

//#include <OpenTissue/core/math/big/big_types.h> // needed for testing export_K
//#include <OpenTissue/core/math/big/io/big_matlab_write.h> // needed for testing export_K



class Application : public OpenTissue::glut::PerspectiveViewApplication
{
public:

  typedef OpenTissue::math::BasicMathTypes<double,size_t>    math_types;
  typedef OpenTissue::fem::Mesh<math_types>                  mesh_type;
  typedef math_types::vector3_type                           vector3_type;
  typedef math_types::real_type                              real_type;

  struct world_point_container
  {
    typedef vector3_type value_type;

    world_point_container(mesh_type * mesh)
      : m_mesh(mesh) 
    {}

    vector3_type & operator[] (unsigned int const & idx)
    {
      return m_mesh->m_nodes[idx].m_coord;
    }

    vector3_type const & operator[] (unsigned int const & idx)const
    {
      return m_mesh->m_nodes[idx].m_coord;
    }

    void clear(){}
    unsigned int size()const{return m_mesh->m_nodes.size();}
    void resize(unsigned int){}

    mesh_type * m_mesh;
  };


  struct original_point_container
  {
    typedef vector3_type value_type;

    original_point_container(mesh_type * mesh) 
      : m_mesh(mesh) 
    {}

    vector3_type & operator[] (unsigned int const & idx)
    {
      return m_mesh->m_nodes[idx].m_model_coord;
    }

    vector3_type const & operator[] (unsigned int const & idx)const
    {
      return m_mesh->m_nodes[idx].m_model_coord;
    }

    void clear(){}

    unsigned int size() const {return m_mesh->m_nodes.size();}

    void resize(unsigned int){}

    mesh_type * m_mesh;
  };

protected:

  bool b[256];          ///< Boolean array used to keep track of the user actions/selections.

  mesh_type     m_mesh1;
  mesh_type     m_mesh2;
  bool          m_stiffness_warp_on;   ///< Boolean value indicating whether stiffness warping is turned on or off.
  real_type     m_gravity;

  void initialize()
  {
		  real_type poisson = 0.33;
        real_type density = 1000;

	  {
	   real_type young = 5000000;
    

          world_point_container point_wrapper(&m_mesh1);
          OpenTissue::t4mesh::generate_blocks(10,3,3,0.1,0.1,0.1,m_mesh1);
          for (int n=0;n<m_mesh1.m_nodes.size();n++)
		  {
			  m_mesh1.m_nodes[n].m_model_coord = m_mesh1.m_nodes[n].m_coord;
			  if (m_mesh1.m_nodes[n].m_model_coord(0) < 0.01)
			  {
				  m_mesh1.m_nodes[n].m_fixed = true;
			  }
		  }

          real_type c_yield = 10e30;     //--- These plasticity settings means that plasticity is turned off
          real_type c_creep = 0;
          real_type c_max = 0;
          OpenTissue::fem::init(m_mesh1,young,poisson,density,c_yield,c_creep,c_max);
        }

		{
			real_type young = 5000000;

          world_point_container point_wrapper(&m_mesh2);
          OpenTissue::t4mesh::generate_blocks(10,3,3,0.1,0.1,0.1,m_mesh2);

		  for (int n=0;n<m_mesh1.m_nodes.size();n++)
		  {
			  m_mesh2.m_nodes[n].m_model_coord = m_mesh2.m_nodes[n].m_coord;
			  if (m_mesh2.m_nodes[n].m_model_coord(0) < 0.01)
			  {
				  m_mesh2.m_nodes[n].m_fixed = true;
			  }
		  }

          //real_type c_yield = 0.1;
          //real_type c_creep = 1;
          //real_type c_max = 1;

          real_type c_yield = .04;  //--- should be less than maximum expected elastic strain in order to see effect (works as a minimum).
          real_type c_creep = .20;  //--- controls how fast the plasticity effect occurs (it is a rate-like control).
          real_type c_max = 0.2;    //--- This is maximum allowed plasticity strain (works as a maximum).
          OpenTissue::fem::init(m_mesh2,young,poisson,density,c_yield,c_creep,c_max);
        }
  }
public:

  Application()
    : m_stiffness_warp_on(true)
    , m_gravity(29.81) 
  {
	  initialize();
  }

public:

  char const * do_get_title() const { return "FEM demo Application"; }

  void do_display()
  {
    OpenTissue::gl::ColorPicker(0.5,0,0);
    OpenTissue::gl::DrawPointsT4Mesh(original_point_container(&m_mesh1),m_mesh1,0.95,true);

    OpenTissue::gl::ColorPicker(0,0.5,0);
    OpenTissue::gl::DrawPointsT4Mesh(world_point_container(&m_mesh1),m_mesh1,0.95,false);

    OpenTissue::gl::ColorPicker(0,0,.5);
    OpenTissue::gl::DrawPointsT4Mesh(world_point_container(&m_mesh2),m_mesh2,0.95,true);
  }

  void do_action(unsigned char choice)
  {
    // Toggle state
    switch(choice)
    {
    case 's':
      m_stiffness_warp_on = !m_stiffness_warp_on;
      std::cout << "Stiffness warping = " << m_stiffness_warp_on << std::endl;
      break;
    case 'g':
      if(m_gravity>0)
      {
        m_gravity = 0;
        std::cout << "Gravity off" << std::endl;
      }
      else
      {
        m_gravity = 9.81;
        std::cout << "Gravity on" << std::endl;
      }
      break;
    case 'i':
      {
       initialize();
      }
      break;

	/*
	case 'x':
      {
        world_point_container point_wrapper(&m_mesh1);

        OpenTissue::t4mesh::xml_write("test.xml",m_mesh1,point_wrapper);
        OpenTissue::t4mesh::xml_read("test.xml",m_mesh1,point_wrapper);
        OpenTissue::fem::update_original_coord(m_mesh1.node_begin(),m_mesh1.node_end());
      }
      break;
	  */
/*
    case 'k':
      {
        using namespace OpenTissue::math::big;

        ublas::compressed_matrix<real_type>  K;
        OpenTissue::fem::export_K(m_mesh1,K);
        std::ofstream file("k_matrix.m");
        file << "K = " << K << std::endl;
        file.flush();
        file.close();
      }
	  */

    default:
      std::cout << "You pressed " << choice << std::endl;
      break;
    };// End Switch
  }

  void do_init_right_click_menu(int main_menu, void menu(int entry))
  {
    int controls = glutCreateMenu(menu);
    glutAddMenuEntry("Inititialize             [i]", 'i');
    glutAddMenuEntry("Toggle Stiffness warping [s]", 's');
    glutAddMenuEntry("Toggle gravity           [g]", 'g');
    glutAddMenuEntry("XML read                 [r]", 'r');
    glutAddMenuEntry("XML write                [w]", 'w');
    glutSetMenu(main_menu);
    glutAddSubMenu("stiffness warping", controls);
  }

  void do_init()
  {
    this->camera().move(95);
  }

  void do_run()
  {

      // Calculate external forces acting on the model.
      // The external forces are stored in each vertex and consists of a downward gravity.
      // If a vertex is being dragged by the user, its velocity vector is added as an external force.
      for (int n=0;n<m_mesh1.m_nodes.size();n++)
	  {
		  m_mesh1.m_nodes[n].m_f_external = vector3_type(0.0, -(m_mesh1.m_nodes[n].m_mass * m_gravity), 0.0);
	  }
      for (int n=0;n<m_mesh2.m_nodes.size();n++)
	  {
		  m_mesh2.m_nodes[n].m_f_external = vector3_type(0.0, -(m_mesh2.m_nodes[n].m_mass * m_gravity), 0.0);
	  }

    OpenTissue::fem::simulate(m_mesh1,0.01,m_stiffness_warp_on,1.0,20,20);
    OpenTissue::fem::simulate(m_mesh2,0.01,m_stiffness_warp_on); 
  }
  void do_shutdown(){}

};



OpenTissue::glut::instance_pointer init_glut_application(int argc, char **argv)
{
  OpenTissue::glut::instance_pointer instance;
  instance= new Application() ;
  return instance;
}

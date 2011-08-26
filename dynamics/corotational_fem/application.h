
// OpenTissue Template Library Demo
// - A specific demonstration of the flexibility of OTTL.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL and OTTL Demos are licensed under zlib.
//
#ifndef FEM_APPLICATION_H
#define FEM_APPLICATION_H


#include <OpenTissue/configuration.h>

#include <MyFemMesh.h>
//#include <OpenTissue/dynamics/fem/fem_mesh.h>

#include <OpenTissue/utility/glut/glut_perspective_view_application.h>





#include <OpenTissue/core/math/math_basic_types.h>
#include <OpenTissue/dynamics/fem/fem.h>



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

//	mesh_type		m_mesh2;
public:
	mesh_type		m_mesh1;

	bool			m_stiffness_warp_on;   ///< Boolean value indicating whether stiffness warping is turned on or off.

	bool			m_collideGroundPlane;

	bool			m_fixNodes;

	real_type		m_gravity;

	real_type		m_young;// = 500000;
	real_type		m_poisson;// = 0.33;
	real_type		m_density;// = 1000;

	//--- infinite m_c_yield plasticity settings means that plasticity is turned off
	real_type		m_c_yield;// = .04;  //--- should be less than maximum expected elastic strain in order to see effect (works as a minimum).
	real_type		m_c_creep;// = .20;  //--- controls how fast the plasticity effect occurs (it is a rate-like control).
	real_type		m_c_max;// = 0.2;    //--- This is maximum allowed plasticity strain (works as a maximum).


  void initialize();
  
public:

  Application()
    : m_stiffness_warp_on(true)
	,m_collideGroundPlane(true)
	,m_fixNodes(false)
    , m_gravity(29.81) 
  {
	  initialize();
  }

public:

  char const * do_get_title() const { return "FEM demo Application"; }

     virtual void mouse_up(double cur_x,double cur_y,bool /*shift*/,bool /*ctrl*/,bool /*alt*/,bool /*left*/,bool /*middle*/,bool /*right*/) ;
		virtual void mouse_down(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right);

       virtual void mouse_move(double cur_x,double cur_y); 
  void do_display();
  
  void do_action(unsigned char choice);

	virtual void reshape();

	void	scaleYoungModulus(float scaling);


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


	void do_init();
  void do_run();
  void do_shutdown();

};




#endif //FEM_APPLICATION_H


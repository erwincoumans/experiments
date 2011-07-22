
// OpenTissue Template Library Demo
// - A specific demonstration of the flexibility of OTTL.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL and OTTL Demos are licensed under zlib.
//
#define DEFINE_GLUT_MAIN
#include <OpenTissue/utility/glut/glut_perspective_view_application.h>
#undef DEFINE_GLUT_MAIN

#include "application.h"
#include "gwenWindow.h"


OpenTissue::glut::instance_pointer init_glut_application(int argc, char **argv)
{
	OpenTissue::glut::instance_pointer instance;
	instance= new Application() ;
	return instance;
}

void Application::initialize()
{
	m_young = 500000;
	m_poisson = 0.33;
	m_density = 1000;

	//--- infinite m_c_yield plasticity settings means that plasticity is turned off
	m_c_yield = .04;  //--- should be less than maximum expected elastic strain in order to see effect (works as a minimum).
	m_c_creep = .20;  //--- controls how fast the plasticity effect occurs (it is a rate-like control).
	m_c_max = 0.2;    //--- This is maximum allowed plasticity strain (works as a maximum).

	{
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
		OpenTissue::fem::init(m_mesh1,m_young,m_poisson,m_density,m_c_yield,m_c_creep,m_c_max);
	}
}



///range 0..1
void	Application::scaleYoungModulus(float scaling)
{
	if (scaling<10)
		scaling=10;
		m_young = 70000*scaling;
		OpenTissue::fem::init(m_mesh1,m_young,m_poisson,m_density,m_c_yield,m_c_creep,m_c_max);
}


void Application::reshape()
{
	if (pCanvas)
		pCanvas->SetSize(width(),height());
}


void Application::mouse_up(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right) 
{
	bool handled = pCanvas->InputMouseButton(0,0);
	if (!handled)
		OpenTissue::glut::PerspectiveViewApplication::mouse_up(cur_x,cur_y,shift,ctrl,alt,left,middle,right);

}
void Application::mouse_down(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right)
{
	bool handled = pCanvas->InputMouseButton(0,1);
	if (!handled)
		OpenTissue::glut::PerspectiveViewApplication::mouse_down(cur_x,cur_y,shift,ctrl,alt,left,middle,right);
}

void Application::mouse_move(double cur_x,double cur_y) 
{
	int x = cur_x;
	int y = cur_y;
	int dx = m_begin_x-cur_x;
	int dy = m_begin_y-cur_y;

	bool handled = false;

	if (pCanvas)
	{
		handled = pCanvas->InputMouseMoved(x,y,dx,dy);
	}

	if (!handled)
		OpenTissue::glut::PerspectiveViewApplication::mouse_move(cur_x,cur_y);
}


void Application::do_display()
{
	OpenTissue::gl::ColorPicker(0.5,0,0);

	OpenTissue::gl::ColorPicker(0.5,0,0);
	OpenTissue::gl::DrawPointsT4Mesh(original_point_container(&m_mesh1),m_mesh1,0.95,true);

	OpenTissue::gl::ColorPicker(0,0.5,0);
	OpenTissue::gl::DrawPointsT4Mesh(world_point_container(&m_mesh1),m_mesh1,0.95,false);



	saveOpenGLState(width(),height());
	pCanvas->RenderCanvas();
	restoreOpenGLState();


}


void Application::do_action(unsigned char choice)
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


void Application::do_init()
{
	static bool once=true;
	if (once)
	{
		once=false;

		setupGUI(this, width(),height()); //will initialize pCanvas etc

	}

	this->camera().move(95);
		OpenTissue::glut::toggleIdle();

}

void Application::do_run()
{

	// Calculate external forces acting on the model.
	// The external forces are stored in each vertex and consists of a downward gravity.
	// If a vertex is being dragged by the user, its velocity vector is added as an external force.
	for (int n=0;n<m_mesh1.m_nodes.size();n++)
	{
		m_mesh1.m_nodes[n].m_f_external = vector3_type(0.0, -(m_mesh1.m_nodes[n].m_mass * m_gravity), 0.0);
	}
	/*for (int n=0;n<m_mesh2.m_nodes.size();n++)
	{
		m_mesh2.m_nodes[n].m_f_external = vector3_type(0.0, -(m_mesh2.m_nodes[n].m_mass * m_gravity), 0.0);
	}
	*/


	OpenTissue::fem::simulate(m_mesh1,0.01,m_stiffness_warp_on,1.0,20,20);
//	OpenTissue::fem::simulate(m_mesh2,0.01,m_stiffness_warp_on); 
}

void Application::do_shutdown()
{
}
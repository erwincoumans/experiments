#ifndef OPENTISSUE_UTILITY_GLUT_GLUT_APPLICATION_H
#define OPENTISSUE_UTILITY_GLUT_GLUT_APPLICATION_H
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
  namespace glut
  {

    /**
     * A base class for any application based on a GLUT binding.
     *
     * This class defines the basic interface of an application. One uses
     * this base class by making specialized derived types. See for instance
     * the class PerspectiveViewApplication as an example.
     *
     * The class also defines data needed for setting up a window mapping
     * (width and height in pixels) and frustum (near and far clipping
     * planes and fov).
     *
     */
    class Application
    {
    protected:

      double          m_fovy;         ///< Field of view in the y-direction.
      double          m_aspect;       ///< Aspect ratio.
      double          m_z_near;       ///< Near clipping plane.
      double          m_z_far;        ///< Far clipping plane.
      int             m_width;        ///< Width of window in pixels.
      int             m_height;       ///< Height of window in pixels.
      int             m_main_window;  ///< Glut handle to main window of the application.
      bool            m_idle_on;      ///< Boolean flag indicating whether the idle function is on or off.

    public:

      int    const & main_window() const { return m_main_window; }
      int          & main_window()       { return m_main_window; }
      int    const & width()       const { return m_width;       }
      int          & width()             { return m_width;       }
      int    const & height()      const { return m_height;      }
      int          & height()            { return m_height;      }
      double const & aspect()      const { return m_aspect;      }
      double       & aspect()            { return m_aspect;      }
      double const & fovy()        const { return m_fovy;        }
      double       & fovy()              { return m_fovy;        }
      double const & z_near()      const { return m_z_near;      }
      double       & z_near()            { return m_z_near;      }
      double const & z_far()       const { return m_z_far;       }
      double       & z_far()             { return m_z_far;       }
      bool   const & idle()        const { return m_idle_on;     }
      bool         & idle()              { return m_idle_on;     }

    public:

      Application()
        : m_fovy(30.0)
        , m_aspect(1.0)
        , m_z_near(0.1)
        , m_z_far(700.0)
        , m_width(640)
        , m_height(480)
        , m_main_window()
        , m_idle_on(false)
      {}

      virtual ~Application(){}

    public:

      virtual char const * get_title() const=0;
      
      virtual void action(unsigned char choice)=0;

      virtual void special_action(int choice)=0;

      virtual void reshape(){};

	  ///range 0..1
	  void	scaleYoungModulus(float scaling);
      
      virtual void init_right_click_menu(int main_menu, void menu(int entry))=0;
      
      virtual void init()=0;
      
      virtual void run()=0;
      
      virtual void shutdown()=0;

      virtual void mouse_down(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right)=0;
      
      virtual void mouse_up(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right)=0;
      
      virtual void mouse_move(double cur_x,double cur_y)=0;

      /**
      * Sets up the modelview transformation prior to invoking the render-method
      * where the actual rendering is supposed to be done.
      *
      * This display handler also contains basic logic for supporting
      * screen capturing and movie making.
      */
      virtual void display()=0;
      
      /**
      * Initializes the openGL state.
      */
      virtual void init_gl_state()=0;
      
      /**
      * Sets up default lighting. It is assumed that the
      * lighting is set up the modelview space (ie. fixed
      * in world).
      *
      */
      virtual void setup_lights()=0;
      
      /**
      * Intenden to Set up a default projection and initialize the modelview
      * transformation. Upon return the modelview matrix stack should be left
      * as the current openGL matrix stack.
      */
      virtual void setup_model_view_projection( )=0;
    };

    /**
    * The application framwork GLUT bindings talk to the application by
    * using a pointer to a Application base class. 
    *
    * End users will create their own applications by making a derived
    * class from the Application and then letting the init_glut_application
    * function return a base-class pointer to an instance of this derived
    * class.
    */
    typedef Application* instance_pointer;

  } // namespace glut
} // namespace OpenTissue


/**
*
* An end user is responsible for actually implementing this function.
*
* End users will create their own applications by making a derived class
* from the Application class and then letting the init_glut_application
* function return a base-class pointer to an instance of this derived class.
*/
OpenTissue::glut::instance_pointer init_glut_application(int argc, char **argv);

/**
* Make sure to define a DEFINE_GLUT_MAIN in the source
* file where you want the application main entry point
* to be.
*
* As an example one could write:
*
*  #define DEFINE_GLUT_MAIN
*  #include #include <OpenTissue/utility/gl/glut_perspective_view_application.h>
*  #undef DEFINE_GLUT_MAIN
*
* In some cpp-file, and just use
*
*  #include #include <OpenTissue/utility/gl/glut_perspective_view_application.h>
*
* In every other file (both header and source files).
*/
#include <OpenTissue/utility/glut/glut_main.ipp>

// OPENTISSUE_UTILITY_GLUT_GLUT_APPLICATION_H
#endif

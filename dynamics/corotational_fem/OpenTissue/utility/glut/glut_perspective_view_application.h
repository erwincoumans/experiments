#ifndef OPENTISSUE_UTILITY_GLUT_GLUT_PERSPECTIVE_VIEW_APPLICATION_H
#define OPENTISSUE_UTILITY_GLUT_GLUT_PERSPECTIVE_VIEW_APPLICATION_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/glut/glut_application.h>




#include <sstream>
#include <iomanip>

namespace OpenTissue
{
  namespace glut
  {

    /**
     * This GLUT application class have been specialized to set up a
     * perspective view mapping. The class also supports basic
     * trackball functionality together with zooming and paning.
     *
     * There is also support for screen captures and movie making.
     *
     * To use this class one simply just create a derived class
     * implementing the pure abstract methods.
     */
    template<
      typename math_types_
    >
    class BasePerspectiveViewApplication : public Application
    {
    public:

      typedef  math_types_  math_types;

    protected:

      OpenTissue::gl::Camera<math_types>	  m_camera;            ///< Camera class taking care of the model-view projection transformation.
      bool                                  m_screen_capture;    ///< Boolean flag indicating whether a screen capture should be performing on the next display event.
      bool						                			m_zoom_mode;         ///< Boolean flag indicating whether the application is currently doing a zoom operation.
      bool									                m_pan_mode;          ///< Boolean flag indicating whether the application is currently doing a pan operation.
      bool									                m_trackball_mode;    ///< Boolean flag indicating whether the application is currently doing a trackball operation.
      double							                  m_begin_x;           ///< The starting x-pixel coordinate when doing a mouse operation.
      double							                  m_begin_y;           ///< The starting y-pixel coordinate when doing a mouse operation.
      double							                  m_zoom_sensitivity;  ///< The zooming mouse sensitivity.
      double							                  m_pan_sensitivity;   ///< The panning mouse sensitivity.

    public:

      OpenTissue::gl::Camera<math_types> const  & camera()           const { return m_camera; }
      OpenTissue::gl::Camera<math_types>        & camera()                 { return m_camera; }
      double const                              & zoom_sensitivity() const { return m_zoom_sensitivity; }
      double                                    & zoom_sensitivity()       { return m_zoom_sensitivity; }
      double const                              & pan_sensitivity()  const { return m_pan_sensitivity; }
      double                                    & pan_sensitivity()        { return m_pan_sensitivity; }

    public:

      BasePerspectiveViewApplication()
        : Application()
        , m_screen_capture(false)
        , m_zoom_mode(false)
        , m_pan_mode(false)
        , m_trackball_mode(false)
        , m_begin_x(0.)
        , m_begin_y(0.)
        , m_zoom_sensitivity(0.25)
        , m_pan_sensitivity(0.25)
      {}

      virtual ~BasePerspectiveViewApplication(){}

    public:

      virtual void do_display()=0;
      virtual void do_action(unsigned char choice)=0;
      virtual void do_special_action(int choice){ /*!EMPTY!*/ };
      virtual void do_init_right_click_menu(int main_menu, void menu(int entry)) = 0;
      virtual void do_init() = 0;
      virtual void do_run() = 0;
      virtual void do_shutdown() = 0;
      virtual char const * do_get_title() const=0;

    public:

      char const * get_title() const { return this->do_get_title(); }

      void run() { this->do_run(); }

      void shutdown() { this->do_shutdown(); }

      void special_action(int choice){ this->do_special_action(choice); }

      void action(unsigned char choice) 
      {
        switch (choice)
        {
        case 'o':
          m_camera.orbit_mode() = !m_camera.orbit_mode();
          if(m_camera.orbit_mode())
            std::cout << "orbit mode on" << std::endl;
          else
            std::cout << "orbit mode off " << std::endl;
          break;
        case 'l':
          m_camera.target_locked() = !m_camera.target_locked();
          if(m_camera.target_locked())
            std::cout << "target is locked" << std::endl;
          else
            std::cout << "target is free " << std::endl;
          break;
        case 'y':
          m_screen_capture = true;
          break;
        default:
          this->do_action(choice);
        };
      }

      void init_right_click_menu(int main_menu, void menu(int entry)) 
      {
        int controls = glutCreateMenu( menu );
        glutAddMenuEntry("quit                    [esc][q]", 'q' );
        glutAddMenuEntry("toggle idle                  [ ]", ' ' );
        glutAddMenuEntry("toggle camera orbit/rotate   [o]", 'o' );
        glutAddMenuEntry("toggle camera target locked  [l]", 'l' );
        glutAddMenuEntry("screen capture               [y]", 'y' );
        glutSetMenu( main_menu );
        glutAddSubMenu( "controls", controls );

        this->do_init_right_click_menu(main_menu, menu);
      }

      void init() 
      {
        //--- setup UI and time
        m_zoom_mode              = false;
        m_pan_mode               = false;
        m_trackball_mode         = false;
        m_camera.target_locked() = true;

        std::cout << "Rotating: hold down left mouse button" << std::endl;
        std::cout << "Zooming: hold down middle mouse button" << std::endl;
        std::cout << "Panning: press shift and hold down left mouse button" << std::endl;

        typedef typename math_types::vector3_type vector3_type;
        vector3_type position = vector3_type(0,0,100);
        vector3_type target   = vector3_type(0,0,0);
        vector3_type up       = vector3_type(0,1,0);

        m_camera.init(position,target,up);

        this->do_init();
      }

      virtual void mouse_down(double cur_x,double cur_y,bool shift,bool ctrl,bool alt,bool left,bool middle,bool right) 
      {
        if (middle || (alt && left))  // 2008-08-13 micky: not all mice have a "normal" middle button!
          m_zoom_mode = true;
        if ( shift && left )
          m_pan_mode = true;
        if(!middle && !right && !ctrl && !alt && !shift && left)// only left button allowed
        {
          m_camera.mouse_down( cur_x, cur_y );
          m_trackball_mode = true;
        }
        m_begin_x = cur_x;
        m_begin_y = cur_y;
      }

      virtual void mouse_up(double cur_x,double cur_y,bool /*shift*/,bool /*ctrl*/,bool /*alt*/,bool /*left*/,bool /*middle*/,bool /*right*/) 
      {
        if ( m_zoom_mode )
        {
          m_camera.move( m_zoom_sensitivity*(cur_y - m_begin_y) );
          m_zoom_mode = false;
        }
        else if ( m_pan_mode )
        {
          m_camera.pan( m_pan_sensitivity*(m_begin_x - cur_x) , m_pan_sensitivity*(cur_y - m_begin_y) );
          m_pan_mode = false;
        }
        else if ( m_trackball_mode )
        {
          m_camera.mouse_up( cur_x, cur_y );
          m_trackball_mode = false;
        }
        m_begin_x = cur_x;
        m_begin_y = cur_y;
      }

      virtual void mouse_move(double cur_x,double cur_y) 
      {
        if ( m_zoom_mode )
        {
          m_camera.move( m_zoom_sensitivity*(cur_y - m_begin_y) );
        }
        else if ( m_pan_mode )
        {
          m_camera.pan( m_pan_sensitivity*(m_begin_x - cur_x) , m_pan_sensitivity*(cur_y - m_begin_y) );
        }
        else if ( m_trackball_mode )
        {
          m_camera.mouse_move( cur_x, cur_y);
        }
        m_begin_x = cur_x;
        m_begin_y = cur_y;
      }

      /**
      * Sets up the modelview transformation prior to invoking the render-method
      * where the actual rendering is supposed to be done.
      *
      * This display handler also contains basic logic for supporting
      * screen capturing and movie making.
      */
      void display()
      {
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        glMatrixMode( GL_MODELVIEW );
        gl::LoadCameraMatrix(m_camera);

        this->do_display();


        glFinish();
        glutSwapBuffers();
      }

      /**
      * Initializes the openGL state.
      * This includes initialization of glew (handles all the extentions)
      *
      *
      */
      void init_gl_state()
      {
        int err = glewInit();
        if ( GLEW_OK != err )
        {
          // problem: glewInit failed, something is seriously wrong
          std::cerr << "GLEW Error: " << glewGetErrorString( err ) << std::endl;
          exit( 1 );
        }
        std::cout << "GLEW status: Using GLEW " << glewGetString( GLEW_VERSION ) << std::endl;

        glClearColor( .7, .7, .7, 1.0 );
        glEnable( GL_DEPTH_TEST );

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        //--- Set up Lights
        glEnable( GL_LIGHTING );
        glEnable( GL_LIGHT0 );
        // light_position is NOT a default value
        // 4th component == 1 means at finite position,
        //               == 0 means at infinity

        // Shading Model
        glShadeModel( GL_SMOOTH );
        glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE );
        glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE );

        this->setup_lights();
      }

      /**
      * Sets up default lighting. It is assumed that the
      * lighting is set up the modelview space (ie. fixed
      * in world).
      *
      */
      void setup_lights()
      {
        // light_position is NOT a default value
        // 4th component == 1 means at finite position,
        //               == 0 means at infinity

        GLfloat AmbientLight[] = { .5, .5, .5, 1.0 };
        GLfloat DiffuseLight[] = { 1., 1., 1., 1.0 };
        GLfloat SpecularLight[] = { 1.0, 1.0, 1.0, 1.0 };

        glLightfv( GL_LIGHT0, GL_AMBIENT, AmbientLight );
        glLightfv( GL_LIGHT0, GL_DIFFUSE, DiffuseLight );
        glLightfv( GL_LIGHT0, GL_SPECULAR, SpecularLight );

        GLfloat LightPosition[] = { 0.0, 0.0, 1.0, 0. };
        glLightfv( GL_LIGHT0, GL_POSITION, LightPosition );
      }

      /**
      * Sets up a default perspective projection and initializes the modelview
      * transformation to the identity transformation. Upon return the modelview
      * matrix stack is left as the current openGL matrix stack.
      *
      */
      void setup_model_view_projection( )
      {
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        this->aspect() = ( 1.0 * this->width() ) / this->height();
        gluPerspective( this->fovy(), this->aspect(), this->z_near(), this->z_far() );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
      }
    };

    typedef BasePerspectiveViewApplication<OpenTissue::math::default_math_types> PerspectiveViewApplication;

  } // namespace glut
} // namespace OpenTissue

// OPENTISSUE_UTILITY_GLUT_GLUT_PERSPECTIVE_VIEW_APPLICATION_H
#endif

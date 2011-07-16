#ifndef OPENTISSUE_UTILITY_GLUT_GLUT_EVENT_HANDLERS_H
#define OPENTISSUE_UTILITY_GLUT_GLUT_EVENT_HANDLERS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

#include <OpenTissue/utility/glut/glut_application.h>

namespace OpenTissue
{
  namespace glut
  {


    instance_pointer application;

    void set_application_instance( OpenTissue::glut::instance_pointer const & ptr) { application = ptr; }

    void mouse( int Button, int State, int cur_x, int cur_y )
    {
      int MODIFIERS = glutGetModifiers();
      bool shift = ( MODIFIERS & GLUT_ACTIVE_SHIFT ) == GLUT_ACTIVE_SHIFT;
      bool alt =   ( MODIFIERS & GLUT_ACTIVE_ALT )   == GLUT_ACTIVE_ALT;
      bool ctrl =  ( MODIFIERS & GLUT_ACTIVE_CTRL )  == GLUT_ACTIVE_CTRL;
      bool left = (Button == GLUT_LEFT_BUTTON);
      bool middle = (Button == GLUT_MIDDLE_BUTTON);
      bool right = (Button == GLUT_RIGHT_BUTTON);
      bool down = (State == GLUT_DOWN);
      if(down)
        application->mouse_down(cur_x,cur_y,shift,ctrl,alt,left,middle,right);
      else
        application->mouse_up(cur_x,cur_y,shift,ctrl,alt,left,middle,right);
      glutPostRedisplay();
    }

    void motion( int cur_x, int cur_y )
    {
      application->mouse_move(cur_x,cur_y);
      glutPostRedisplay();
    }

    void reshape( int cur_width, int cur_height )
    {
      application->width() = cur_width;
      application->height() = cur_height;

      glViewport( 0, 0, application->width(), application->height() );

      application->setup_model_view_projection();
      application->setup_lights();

      glutPostRedisplay();
    }

    void idle()
    {
      application->run();
      glutPostRedisplay();
    }

    void key( unsigned char k, int /*x*/, int /*y*/ )
    {
      switch (k)
      {
      case 27:
      case 'q':
        application->shutdown();

        exit(k);
        break;
      case ' ':
        application->idle() = !application->idle();
        if(application->idle())
          glutIdleFunc( &idle  );
        else
          glutIdleFunc(0);
        //break;  // [micky] No, (some) applications actually want to be notified about this one!!!
      default:
        application->action(k);
        break;
      };
      glutPostRedisplay();
    }

    void specialkey(int k, int /*x*/, int /*y*/)
    {
      switch (k){
        case -1  :
          break;
        default:
          application->special_action(k);
          break;
      };
      glutPostRedisplay();
      return;
    }

    void display()  
    {
      application->display();  
    }

    void menu( int entry )
    {
      key( entry, 0, 0 );
    }

  } // namespace glut
} // namespace OpenTissue

// OPENTISSUE_UTILITY_GLUT_GLUT_EVENT_HANDLERS_H
#endif

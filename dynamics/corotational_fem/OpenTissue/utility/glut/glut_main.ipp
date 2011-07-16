#ifdef DEFINE_GLUT_MAIN
//
// OpenTissue, A toolbox for physical based simulation and animation.
// Copyright (C) 2007 Department of Computer Science, University of Copenhagen
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>

#include <OpenTissue/utility/glut/glut_application.h>
#include <OpenTissue/utility/glut/glut_event_handlers.h>


int main( int argc, char **argv )
{
  glutInit( &argc, argv );

  OpenTissue::glut::instance_pointer application = init_glut_application(argc,argv);

  OpenTissue::glut::set_application_instance(application);

  glutInitDisplayMode( GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE );
  glutInitWindowSize( application->width(), application->height() );
  glutInitWindowPosition( 50, 50 );
  application->main_window() = glutCreateWindow( application->get_title() );

  application->init_gl_state();

  int main_menu = glutCreateMenu( &OpenTissue::glut::menu );

  application->init_right_click_menu( main_menu, &OpenTissue::glut::menu );

  glutSetMenu( main_menu );
  glutAttachMenu( GLUT_RIGHT_BUTTON );

  application->init();
    
  glutDisplayFunc ( &OpenTissue::glut::display );
  glutReshapeFunc ( &OpenTissue::glut::reshape );
  glutMouseFunc   ( &OpenTissue::glut::mouse   );
  glutMotionFunc  ( &OpenTissue::glut::motion  );
  glutPassiveMotionFunc( &OpenTissue::glut::motion  );
  glutKeyboardFunc( &OpenTissue::glut::key     );
  glutSpecialFunc ( &OpenTissue::glut::specialkey );  

  glutMainLoop();

  application->shutdown();

  return 0;
}

#endif // DEFINE_GLUT_MAIN

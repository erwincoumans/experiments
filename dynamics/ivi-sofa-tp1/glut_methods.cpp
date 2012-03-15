#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

#include <iostream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "simulation.h"
#include "render.h"

void reshape(int w, int h);
void display();
void idle();
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void close();
void menu(int item);

//// GLUT Methods ////

int glut_start_step = 0;

int window_width = 800;
int window_height = 600;

int pause_animation = 0;
int camera_mode = -1;
int picking_mode = -1;
double fps = 0;
double mean_fps = 0;

int winid = 0;
int menuid = 0;

const char* title_loading = SOFA_DEVICE " FEM Demo [loading...]";
const char* title_active = SOFA_DEVICE " FEM Demo [running]";
const char* title_paused = SOFA_DEVICE " FEM Demo [paused]";

void init_glut(int* argc, char** argv)
{
    if (glut_start_step > 0) return;
    glutInit(argc, argv);
    glutInitWindowSize(window_width, window_height);
#if (GLUT_API_VERSION >= 4 || GLUT_XLIB_IMPLEMENTATION >= 9)
    glutInitDisplayString("rgba depth>=16 double samples");
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);
#endif
    winid = glutCreateWindow(title_loading);
    glutReshapeFunc ( reshape );
    glutDisplayFunc ( display );
	glutIdleFunc ( idle );
    glutMouseFunc ( mouse );
    glutMotionFunc ( motion );
    glutKeyboardFunc ( keyboard );
    glutSpecialFunc ( special );
    //glutWMCloseFunc ( close );
/*
    menuid = glutCreateMenu(menu);
    glutAddMenuEntry("test",1);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
*/
    glewInit();

    glMatrixMode   ( GL_MODELVIEW );  // Select The Model View Matrix
    glLoadIdentity ( );    // Reset The Model View Matrix

    glClearColor ( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();

    glut_start_step = 1;
}

int main_load();

void setup_glut()
{
    if (glut_start_step != 1) return;
    if (main_load())
    {
        exit(1);
    }

    glut_start_step = 2;
	initgl();
    reshape(window_width, window_height);
    glutSetWindowTitle(pause_animation ? title_paused : title_active);

}

void run_glut()
{
    glutMainLoop();
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    glMatrixMode   ( GL_PROJECTION );  // Select The Projection Matrix
    glLoadIdentity ( );                // Reset The Projection Matrix
    gluPerspective ( 40, (GLdouble)w / (GLdouble)h, 0.001*simulation_size, 100.0*simulation_size );
    glMatrixMode   ( GL_MODELVIEW );  // Select The Model View Matrix
}

void display()
{
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if (glut_start_step >= 2)
    {
        render();
        render_gui();
    }
    glutSwapBuffers();
}

enum { FPS_ITERATIONS=100 };
enum { FPS_SAMPLES=10 };
double iter_time_buffer[FPS_SAMPLES];
int iter_last_time = 0;
int anim_iteration = 0;
int pause_animation_time = 0;
extern bool render_surface;
int nb_idle = 0;
void idle()
{
    ++nb_idle;
    if (glut_start_step == 1)
    {
        if (nb_idle < 10)
        {
            glutPostRedisplay();
        }
        setup_glut();
    }
    if (glut_start_step < 2) return;

    if (!pause_animation)
    {
        if (anim_iteration % FPS_ITERATIONS == 0)
        {
            int t = glutGet(GLUT_ELAPSED_TIME);
            //std::cout << t * 0.001 << std::endl;
            if (anim_iteration > 0)
            {
                double dt = t - iter_last_time;
                fps = (FPS_ITERATIONS*1000) / dt;
                int s = (anim_iteration/FPS_ITERATIONS);
                iter_time_buffer[(s-1) % FPS_SAMPLES] = dt;
                int ns = (s >= FPS_SAMPLES) ? FPS_SAMPLES : s;
                double ttotal = 0;
                for (int i = s-ns; i < s; ++i)
                    ttotal += iter_time_buffer[i % FPS_SAMPLES];
                mean_fps = (ns * FPS_ITERATIONS * 1000) / ttotal;
                std::cout << "fps = " << fps;
                if (ns >= FPS_SAMPLES)
                    std::cout << "\t mean of last " << ns << " fps = " << mean_fps;
                std::cout << std::endl;
            }
            iter_last_time = t;
        }
        if (picking_mode != -1)
            picking_mode = update_picking(picking_mode);
        simulation_animate();
        if (render_surface) // if the surfaces are currently rendered, update them now
            simulation_mapping();

        ++anim_iteration;
    }
	glutPostRedisplay();
}

#if !defined(GLUT_WHEEL_UP)
#  define GLUT_WHEEL_UP   3
#  define GLUT_WHEEL_DOWN 4
#endif

int mouse_btstate = 0;
int mouse_x = 0;
int mouse_y = 0;

void motion(int x, int y)
{
    if (glut_start_step < 2) return;

    int dx = (mouse_btstate) ? x - mouse_x : 0;
    int dy = (mouse_btstate) ? y - mouse_y : 0;
    mouse_x = x;
    mouse_y = y;
    if (camera_mode >= 0)
    {
        move_camera(dx, dy, camera_mode);
        glutPostRedisplay();
    }
    if (picking_mode >= 0 && !pause_animation)
    {
        double viewmatrix[16];
        glGetDoublev(GL_MODELVIEW_MATRIX,viewmatrix);
        double projmatrix[16];
        glGetDoublev(GL_PROJECTION_MATRIX,projmatrix);
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT,viewport);
        double obj_x = 0, obj_y = 0, obj_z = 0;
        gluUnProject(x, viewport[3]-y, 0, viewmatrix, projmatrix, viewport, &obj_x, &obj_y, &obj_z);
        picking_mode = move_picking(picking_mode, x, y, dx, dy, obj_x, obj_y, obj_z);
        //if (pause_animation) glutPostRedisplay();
    }
}

void mouse(int button, int state, int x, int y)
{
    if (glut_start_step < 2) return;

    motion(x,y);
    int modifiers = glutGetModifiers();
    int bt = -1;
    switch(button)
    {
    case GLUT_LEFT_BUTTON : bt = 0; break;
    case GLUT_RIGHT_BUTTON : bt = 1; break;
    case GLUT_MIDDLE_BUTTON : bt = 2; break;
    case GLUT_WHEEL_UP:
        move_camera(0,10,2);
        glutPostRedisplay();
        break;
    case GLUT_WHEEL_DOWN:
        move_camera(0,-10,2);
        glutPostRedisplay();
        break;
    }
    if (bt >= 0)
        if (state == GLUT_DOWN)
            mouse_btstate |= (1<<bt);
        else
            mouse_btstate &= ~(1<<bt);
    // camera motion
    if (state == GLUT_DOWN && (modifiers == GLUT_ACTIVE_SHIFT || (pause_animation && modifiers == 0)))
    {
        camera_mode = bt;
    }
    else
    {
        camera_mode = -1;
    }
    if (camera_mode == -1 && /*state == GLUT_DOWN &&*/ modifiers == 0 && !pause_animation)
    { // picking
        double viewmatrix[16];
        glGetDoublev(GL_MODELVIEW_MATRIX,viewmatrix);
        double projmatrix[16];
        glGetDoublev(GL_PROJECTION_MATRIX,projmatrix);
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT,viewport);
        double obj_x = 0, obj_y = 0, obj_z = 0;
        gluUnProject(x, viewport[3]-y, 0, viewmatrix, projmatrix, viewport, &obj_x, &obj_y, &obj_z);
        if (picking_mode == -1 && state == GLUT_DOWN)
        {
            picking_mode = start_picking(picking_mode, x, y, bt, obj_x, obj_y, obj_z);
            if (picking_mode == -1) // move camera instead
                camera_mode = bt;
        }
        else if (picking_mode != -1)
            picking_mode = stop_picking(picking_mode, x, y, bt, obj_x, obj_y, obj_z);
    }
}

int fullscreen = 0;
int prev_width = 800;
int prev_height = 600;
int prev_win_x = 0;
int prev_win_y = 0;

void keyboard(unsigned char key, int x, int y)
{
    if (glut_start_step < 2 && key != 27) return;

    int modifiers = glutGetModifiers();
    switch (key)
    {
    case ' ': // SPACE
    {
        pause_animation = !pause_animation;
        int t = glutGet(GLUT_ELAPSED_TIME);
        if (pause_animation)
            pause_animation_time = t;
        else if (anim_iteration > 0)
            iter_last_time += t - pause_animation_time;
        glutSetWindowTitle(pause_animation ? title_paused : title_active);
        glutIdleFunc ( (pause_animation) ? NULL : idle );
        break;
    }
    case 8: // DEL
    case 127: // SUPPR
        simulation_reset();
        if (pause_animation) glutPostRedisplay();
        break;
    case '0':
        reset_camera();
        if (pause_animation) glutPostRedisplay();
        break;
    case 27: // ESC
        if (!fullscreen)
        {
            exit(0);
            //glutDestroyWindow(glutGetWindow());
            break;
        }
    case '\r': // ENTER
        fullscreen = !fullscreen;
        if (fullscreen)
        {
            prev_win_x = glutGet(GLUT_WINDOW_X);
            prev_win_y = glutGet(GLUT_WINDOW_Y);
            prev_width = glutGet(GLUT_WINDOW_WIDTH);
            prev_height = glutGet(GLUT_WINDOW_HEIGHT);
            glutFullScreen();
        }
        else
        {
            glutReshapeWindow(prev_width,  prev_height);
            glutPositionWindow(prev_win_x, prev_win_y);
        }
        break;
    case 'r':
    case 'R':
        shaders_reload();
        if (pause_animation) glutPostRedisplay();
        break;
    case 's':
    case 'S':
        simulation_save();
        if (pause_animation) glutPostRedisplay();
        break;
    case 'l':
    case 'L':
        simulation_load();
        if (pause_animation) glutPostRedisplay();
        break;
    default:
        std::cout << "key 0x"<<std::hex<<(int)key<<std::dec;
        if (key >= 32 && key < 127)
            std::cout << " '"<<(char)key << "'";
        std::cout << " pressed." << std::endl;
    }
}

void special(int key, int x, int y)
{
    if (glut_start_step < 2) return;

    int modifiers = glutGetModifiers();
    if (key >= GLUT_KEY_F1 && key < GLUT_KEY_F1 + RENDER_NFLAGS)
    {
        ++render_flag[key-GLUT_KEY_F1];
        glutPostRedisplay();
        return;
    }
    switch (key)
    {
    case GLUT_KEY_UP:
        move_camera(0,10,2);
        glutPostRedisplay();
        break;
    case GLUT_KEY_DOWN:
        move_camera(0,-10,2);
        glutPostRedisplay();
        break;
    case GLUT_KEY_LEFT:
        move_camera( 50,0,1);
        glutPostRedisplay();
        break;
    case GLUT_KEY_RIGHT:
        move_camera(-50,0,1);
        glutPostRedisplay();
        break;
    case GLUT_KEY_PAGE_UP:
        move_camera(0,25,1);
        glutPostRedisplay();
        break;
    case GLUT_KEY_PAGE_DOWN:
        move_camera(0,-25,1);
        glutPostRedisplay();
        break;
    default:
        std::cout << "special key 0x"<<std::hex<<(int)key<<std::dec;
        //if (key >= 32 && key < 127)
        //    std::cout << " '"<<(char)key << "'";
        std::cout << " pressed." << std::endl;
    }
}

void menu(int item)
{
    std::cout << "menu " << item << std::endl;
}

void close()
{
    std::cout << "closing" << std::endl;
}

float drawText_x = 0;
float drawText_x0 = 0;
float drawText_y = 0;
void* drawText_glutFont = GLUT_BITMAP_HELVETICA_12;
float drawText_lineH = 15;
float drawText_alignH = 0;
float drawText_alignV = 0;

void drawTextPos(float x, float y)
{
    drawText_x0 = drawText_x = x;
    drawText_y = y;
    //glRasterPos2i(x, window_height-y);
}

void drawTextAlign(float h, float v)
{
    drawText_alignH = h;
    drawText_alignV = v;
}

void drawTextFont(float size, bool serif)
{
    if (serif)
    {
        if (size < 17)
        { drawText_glutFont = GLUT_BITMAP_TIMES_ROMAN_10; drawText_lineH = 10; }
        else
        { drawText_glutFont = GLUT_BITMAP_TIMES_ROMAN_24; drawText_lineH = 24; }
    }
    else
    {
        if (size < 12)
        { drawText_glutFont = GLUT_BITMAP_HELVETICA_10; drawText_lineH = 10; }
        else if (size < 16)
        { drawText_glutFont = GLUT_BITMAP_HELVETICA_12; drawText_lineH = 12; }
        else
        { drawText_glutFont = GLUT_BITMAP_HELVETICA_18; drawText_lineH = 18; }
    }
    if (size > drawText_lineH) drawText_lineH = size;
    drawText_lineH += 2; // line spacing
}

float drawText(const char* str)
{
    float x = drawText_x;
    float y = drawText_y;
    float h = 0;
    float lineH = drawText_lineH;
    if (drawText_alignV != 0)
    {
        int nbl = 0;
        const char* s;
        for (s = str; *s; ++s)
            if (*s == '\n') ++nbl;
        if (s > str && s[-1] != '\n') ++nbl;
        y -= lineH*nbl * drawText_alignV;
        if (drawText_alignV <= 0.5f)
            drawText_y += lineH*nbl;
        else
            drawText_y -= lineH*nbl;
        //glRasterPos2i(x, window_height-y);
    }

    bool eol = true;
    while (*str)
    {
        char c = *str;
        if (eol && drawText_alignH != 0)
        {
            float w = 0;
            const char* s;
            for (s = str+1; *s && *s != '\n'; ++s)
                w += glutBitmapWidth(drawText_glutFont, *s);
            x = drawText_x0 - w * drawText_alignH;
        }
        eol = (c == '\n');
        if (eol)
        {
            y += lineH; h += lineH;
            x = drawText_x0;
        }
        else
        {
            glRasterPos2i(x, window_height-y);
            glutBitmapCharacter(drawText_glutFont, c);
            x += glutBitmapWidth(drawText_glutFont, c);
        }
        ++str;
    }
    if ((drawText_alignV != 0 || drawText_alignH == 0) && !eol)
    {
        y += lineH; h += lineH; // auto end-of-line
        x = drawText_x0;
    }
    if (drawText_alignV == 0)
        drawText_y = y;
    if (drawText_alignH == 0)
        drawText_x = x;

    return h;
}

float drawTextF(const char* fmt, ...)
{
	va_list args;
	va_start( args, fmt );
    char buffer[2048];
	vsnprintf(buffer, sizeof(buffer), fmt, args );
	va_end( args );
    // make sure the string is zero-terminated
    buffer[sizeof(buffer)-1] = '\0';
	return drawText(buffer);
}

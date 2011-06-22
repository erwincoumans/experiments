
///from http://wiki.wxwidgets.org/WxGLCanvas

#include "wx/wx.h"
#include "wx/sizer.h"
#include "wx/glcanvas.h"
#include "main.h"
#include "MyFrame.h"

// include OpenGL
#ifdef __WXMAC__
#include "OpenGL/glu.h"
#include "OpenGL/gl.h"
#else
#include <GL/glu.h>
#include <GL/gl.h>
#endif
 

class MyApp: public wxApp
{
    virtual bool OnInit();
    
    wxFrame *frame;
    BasicGLPane * glPane;
public:
    
};
 
IMPLEMENT_APP(MyApp)
 
 
bool MyApp::OnInit()
{
    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
    //frame = new wxFrame((wxFrame *)NULL, -1,  wxT("Hello GL World"), wxPoint(50,50), wxSize(400,200));
	frame = new MyFrame((wxFrame *)NULL,  wxT("Hello GL World"));
	
	
    int args[] = {WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 16, 0};
    
    glPane = new BasicGLPane( (wxFrame*) frame, args);
    sizer->Add(glPane, 1, wxEXPAND);
	
    frame->SetSizer(sizer);
    frame->SetAutoLayout(true);
	frame->Maximize(true);
	frame->SetSize(0,0,800,600);
	
    frame->Show();
    return true;
} 
 
BEGIN_EVENT_TABLE(BasicGLPane, wxGLCanvas)
EVT_MOTION(BasicGLPane::mouseMoved)
EVT_LEFT_DOWN(BasicGLPane::mouseDown)
EVT_LEFT_UP(BasicGLPane::mouseReleased)
EVT_RIGHT_DOWN(BasicGLPane::rightClick)
EVT_LEAVE_WINDOW(BasicGLPane::mouseLeftWindow)
EVT_SIZE(BasicGLPane::resized)
EVT_KEY_DOWN(BasicGLPane::keyPressed)
EVT_KEY_UP(BasicGLPane::keyReleased)
EVT_MOUSEWHEEL(BasicGLPane::mouseWheelMoved)
EVT_PAINT(BasicGLPane::render)
END_EVENT_TABLE()
 
 
// some useful events to use
void BasicGLPane::mouseMoved(wxMouseEvent& event) {}
void BasicGLPane::mouseDown(wxMouseEvent& event) {}
void BasicGLPane::mouseWheelMoved(wxMouseEvent& event) {}
void BasicGLPane::mouseReleased(wxMouseEvent& event) {}
void BasicGLPane::rightClick(wxMouseEvent& event) {}
void BasicGLPane::mouseLeftWindow(wxMouseEvent& event) {}
void BasicGLPane::keyPressed(wxKeyEvent& event) {}
void BasicGLPane::keyReleased(wxKeyEvent& event) {}
 
// Vertices and faces of a simple cube to demonstrate 3D render
// source: http://www.opengl.org/resources/code/samples/glut_examples/examples/cube.c
GLfloat v[8][3];
GLint faces[6][4] = {  /* Vertex indices for the 6 faces of a cube. */
    {0, 1, 2, 3}, {3, 2, 6, 7}, {7, 6, 5, 4},
    {4, 5, 1, 0}, {5, 6, 2, 1}, {7, 4, 0, 3} };
 


BasicGLPane::BasicGLPane(wxFrame* parent, int* args) :
    wxGLCanvas(parent, wxID_ANY,  0, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE)
{
	m_context = new wxGLContext(this);
    // prepare a simple cube to demonstrate 3D render
    // source: http://www.opengl.org/resources/code/samples/glut_examples/examples/cube.c
    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -1;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = 1;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -1;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = 1;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = 1;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = -1;    
}
 
BasicGLPane::~BasicGLPane()
{
	delete m_context;
}

void BasicGLPane::resized(wxSizeEvent& evt)
{
//	wxGLCanvas::OnSize(evt);
	
    Refresh();
}
 
/** Inits the OpenGL viewport for drawing in 3D. */
void BasicGLPane::prepare3DViewport(int topleft_x, int topleft_y, int bottomrigth_x, int bottomrigth_y)
{
	
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black Background
    glClearDepth(1.0f);	// Depth Buffer Setup
    glEnable(GL_DEPTH_TEST); // Enables Depth Testing
    glDepthFunc(GL_LEQUAL); // The Type Of Depth Testing To Do
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	
    glEnable(GL_COLOR_MATERIAL);
	
    glViewport(topleft_x, topleft_y, bottomrigth_x-topleft_x, bottomrigth_y-topleft_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	
    float ratio_w_h = (float)(bottomrigth_x-topleft_x)/(float)(bottomrigth_y-topleft_y);
    gluPerspective(45 /*view angle*/, ratio_w_h, 0.1 /*clip close*/, 200 /*clip far*/);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
}
 
/** Inits the OpenGL viewport for drawing in 2D. */
void BasicGLPane::prepare2DViewport(int topleft_x, int topleft_y, int bottomrigth_x, int bottomrigth_y)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black Background
    glEnable(GL_TEXTURE_2D);   // textures
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	
    glViewport(topleft_x, topleft_y, bottomrigth_x-topleft_x, bottomrigth_y-topleft_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    gluOrtho2D(topleft_x, bottomrigth_x, bottomrigth_y, topleft_y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
 
int BasicGLPane::getWidth()
{
    return GetSize().x;
}
 
int BasicGLPane::getHeight()
{
    return GetSize().y;
}
 
 
void BasicGLPane::render( wxPaintEvent& evt )
{
    if(!IsShown()) return;
    
    wxGLCanvas::SetCurrent(*m_context);
    wxPaintDC(this); // only to be used in paint events. use wxClientDC to paint outside the paint event
	
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    // ------------- draw some 2D ----------------
    prepare2DViewport(0,0,getWidth()/2, getHeight());
    glLoadIdentity();
	
    // white background
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glVertex3f(0,0,0);
    glVertex3f(getWidth(),0,0);
    glVertex3f(getWidth(),getHeight(),0);
    glVertex3f(0,getHeight(),0);
    glEnd();
	
    // red square
    glColor4f(1, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex3f(getWidth()/8, getHeight()/3, 0);
    glVertex3f(getWidth()*3/8, getHeight()/3, 0);
    glVertex3f(getWidth()*3/8, getHeight()*2/3, 0);
    glVertex3f(getWidth()/8, getHeight()*2/3, 0);
    glEnd();
    
    // ------------- draw some 3D ----------------
    prepare3DViewport(getWidth()/2,0,getWidth(), getHeight());
    glLoadIdentity();
	
    glColor4f(0,0,1,1);
    glTranslatef(0,0,-5);
    glRotatef(50.0f, 0.0f, 1.0f, 0.0f);
    
    glColor4f(1, 0, 0, 1);
    for (int i = 0; i < 6; i++)
    {
        glBegin(GL_LINE_STRIP);
        glVertex3fv(&v[faces[i][0]][0]);
        glVertex3fv(&v[faces[i][1]][0]);
        glVertex3fv(&v[faces[i][2]][0]);
        glVertex3fv(&v[faces[i][3]][0]);
        glVertex3fv(&v[faces[i][0]][0]);
        glEnd();
    }
    
    glFlush();
    SwapBuffers();
}

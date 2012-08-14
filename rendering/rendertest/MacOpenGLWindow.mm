#include "MacOpenGLWindow.h"


#import <Cocoa/Cocoa.h>
#include <OpenGL/gl3.h>

#include <OpenGL/glext.h>

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>


extern bool pauseSimulation;
extern bool shootObject;
extern int m_glutScreenWidth;
extern int m_glutScreenHeight;

/* report GL errors, if any, to stderr */
static void checkError(const char *functionName)
{
    GLenum error;
    while (( error = glGetError() ) != GL_NO_ERROR)
    {
        fprintf (stderr, "GL error 0x%X detected in %s\n", error, functionName);
    }
}

void dumpInfo(void)
{
    printf ("Vendor: %s\n", glGetString (GL_VENDOR));
    printf ("Renderer: %s\n", glGetString (GL_RENDERER));
    printf ("Version: %s\n", glGetString (GL_VERSION));
    printf ("GLSL: %s\n", glGetString (GL_SHADING_LANGUAGE_VERSION));
    checkError ("dumpInfo");
}





// -------------------- View ------------------------

@interface TestView : NSView 
{ 
    NSOpenGLContext* m_context;
    int m_lastWidth;
    int m_lastHeight;
}
-(void)drawRect:(NSRect)rect;
-(void) MakeContext:(NSView*) view;
-(void) MakeCurrent;
-(float) GetWindowWidth;
-(float) GetWindowHeight;

@end

float loop;

#define Pi 3.1415

@implementation TestView

-(float) GetWindowWidth
{
    return m_lastWidth;
}
-(float) GetWindowHeight
{
    return m_lastHeight;
}

-(void)drawRect:(NSRect)rect
{
	if (([self frame].size.width != m_lastWidth) || ([self frame].size.height != m_lastHeight))
	{
		m_lastWidth = [self frame].size.width;
		m_lastHeight = [self frame].size.height;
		
		// Only needed on resize:
		[m_context clearDrawable];
		
//		reshape([self frame].size.width, [self frame].size.height);
        float width = [self frame].size.width;
        float height = [self frame].size.height;
        
        
        // Get view dimensions in pixels
        NSRect backingBounds = [self convertRectToBacking:[self bounds]];
        
        GLsizei backingPixelWidth  = (GLsizei)(backingBounds.size.width),
        backingPixelHeight = (GLsizei)(backingBounds.size.height);
        
        // Set viewport
       // glViewport(0, 0, backingPixelWidth, backingPixelHeight);
     //   glViewport(0,0,10,10);
        
        
        //glViewport(0,0,(GLsizei)width,(GLsizei)height);

	}
	
	[m_context setView: self];
	[m_context makeCurrentContext];
	
	// Draw
	//display();
	
	[m_context flushBuffer];
	[NSOpenGLContext clearCurrentContext];
	
	loop = loop + 0.1;
}

-(void) MakeContext
{
    //	NSWindow *w;
	NSOpenGLPixelFormat *fmt;
    
	NSOpenGLPixelFormatAttribute attrs[] =
	{
		NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
		NSOpenGLPFADoubleBuffer,
		NSOpenGLPFADepthSize, 32,
		(NSOpenGLPixelFormatAttribute)0
	};
        
	// Init GL context
	fmt = [[NSOpenGLPixelFormat alloc] initWithAttributes: (NSOpenGLPixelFormatAttribute*)attrs];
	
	m_context = [[NSOpenGLContext alloc] initWithFormat: fmt shareContext: nil];
	[fmt release];
	[m_context makeCurrentContext];
    
	checkError("makeCurrentContext");
}

-(void) MakeCurrent
{
    [m_context makeCurrentContext];
}
-(void)windowWillClose:(NSNotification *)note
{
    [[NSApplication sharedApplication] terminate:self];
}
@end

struct MacOpenGLWindowInternalData
{
    MacOpenGLWindowInternalData()
    {
        m_myApp = 0;
        m_myview = 0;
        m_pool = 0;
        m_window = 0;
        m_width = -1;
        m_height = -1;
        m_exitRequested = false;
    }
    NSApplication*      m_myApp;
    TestView*             m_myview;
    NSAutoreleasePool*  m_pool;
    NSWindow*           m_window;
    int m_width;
    int m_height;
    bool m_exitRequested;
    
};

MacOpenGLWindow::MacOpenGLWindow()
:m_internalData(0),
m_mouseX(0),
m_mouseY(0),
m_mouseCallback(0),
m_keyboardCallback(0)
{
}

MacOpenGLWindow::~MacOpenGLWindow()
{
    if (m_internalData)
        exit();
}



void MacOpenGLWindow::init(int width, int height)
{
    if (m_internalData)
        exit();
    
    
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    
    m_internalData = new MacOpenGLWindowInternalData;
    m_internalData->m_width = width;
    m_internalData->m_height = height;
    
    m_internalData->m_pool = [NSAutoreleasePool new];
	m_internalData->m_myApp = [NSApplication sharedApplication];
	//myApp = [MyApp sharedApplication];
	//home();
    
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    
    id menubar = [[NSMenu new] autorelease];
    id appMenuItem = [[NSMenuItem new] autorelease];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];
    
    id appMenu = [[NSMenu new] autorelease];
    id appName = [[NSProcessInfo processInfo] processName];
    id quitTitle = [@"Quit " stringByAppendingString:appName];
    id quitMenuItem = [[[NSMenuItem alloc] initWithTitle:quitTitle
                                                  action:@selector(terminate:) keyEquivalent:@"q"] autorelease];
    
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];
 
    NSMenuItem *fileMenuItem = [[NSMenuItem new] autorelease];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenuItem setSubmenu: fileMenu]; // was setMenu:
    
    NSMenuItem *newMenu = [[NSMenuItem alloc] initWithTitle:@"New" action:NULL keyEquivalent:@""];
    NSMenuItem *openMenu = [[NSMenuItem alloc] initWithTitle:@"Open" action:NULL keyEquivalent:@""];
    NSMenuItem *saveMenu = [[NSMenuItem alloc] initWithTitle:@"Save" action:NULL keyEquivalent:@""];

    [fileMenu addItem: newMenu];
    [fileMenu addItem: openMenu];
    [fileMenu addItem: saveMenu];
    [menubar addItem: fileMenuItem];
        
    
    // add Edit menu
    NSMenuItem *editMenuItem = [[NSMenuItem new] autorelease];
    NSMenu *menu = [[NSMenu allocWithZone:[NSMenu menuZone]]initWithTitle:@"Edit"];
    [editMenuItem setSubmenu: menu];
    
    NSMenuItem *copyItem = [[NSMenuItem allocWithZone:[NSMenu menuZone]]initWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    
    [menu addItem:copyItem];
    [menubar addItem:editMenuItem];
    
   // [mainMenu setSubmenu:menu forItem:menuItem];
    
    
    //NSMenuItem *fileMenuItem = [[NSMenuItem alloc] initWithTitle: @"File"];
    /*[fileMenuItem setSubmenu: fileMenu]; // was setMenu:
    [fileMenuItem release];
    */
    
    /*NSMenu *newMenu;
    NSMenuItem *newItem;
    
    // Add the submenu
    newItem = [[NSMenuItem allocWithZone:[NSMenu menuZone]]
               initWithTitle:@"Flashy" action:NULL keyEquivalent:@""];
    newMenu = [[NSMenu allocWithZone:[NSMenu menuZone]]
               initWithTitle:@"Flashy"];
    [newItem setSubmenu:newMenu];
    [newMenu release];
    [[NSApp mainMenu] addItem:newItem];
    [newItem release];
    */
    
	NSRect frame = NSMakeRect(0., 0., width, height);
	
	m_internalData->m_window = [NSWindow alloc];
	[m_internalData->m_window initWithContentRect:frame
                      styleMask:NSTitledWindowMask |NSResizableWindowMask| NSClosableWindowMask | NSMiniaturizableWindowMask
                        backing:NSBackingStoreBuffered
                          defer:false];
	[m_internalData->m_window setTitle:@"Minimal OpenGL app (Cocoa)"];
    
	m_internalData->m_myview = [TestView alloc];
	[m_internalData->m_myview initWithFrame: frame];
    
	// OpenGL init!
	[m_internalData->m_myview MakeContext];

   // https://developer.apple.com/library/mac/#documentation/GraphicsAnimation/Conceptual/HighResolutionOSX/CapturingScreenContents/CapturingScreenContents.html#//apple_ref/doc/uid/TP40012302-CH10-SW1
    //support HighResolutionOSX
   [m_internalData->m_myview  setWantsBestResolutionOpenGLSurface:YES];

    
    dumpInfo();
    

    
 
    [m_internalData->m_window setContentView: m_internalData->m_myview];

    GLuint n = 1;
    GLuint               vbo[3]={-1,-1,-1};
    
	glGenBuffers(n, vbo);
    checkError("glGenBuffers");

    
	[m_internalData->m_window setDelegate:(id) m_internalData->m_myview];
	glGenBuffers(n, vbo);
    checkError("glGenBuffers");

    [m_internalData->m_window makeKeyAndOrderFront: nil];
    
    [m_internalData->m_myview MakeCurrent];
    
    glGenBuffers(n, vbo);
    checkError("glGenBuffers");
    
    [NSApp activateIgnoringOtherApps:YES];
    glGenBuffers(n, vbo);
    checkError("glGenBuffers");
    
    //see http://stackoverflow.com/questions/8238473/cant-get-nsmousemoved-events-from-nexteventmatchingmask-with-an-nsopenglview
   /*
    ProcessSerialNumber psn;
    GetCurrentProcess(&psn);
    TransformProcessType(&psn, kProcessTransformToForegroundApplication);
    SetFrontProcess(&psn);
    
    [NSEvent addGlobalMonitorForEventsMatchingMask:NSMouseMovedMask];
    */
    
     [m_internalData->m_myApp finishLaunching];
    [pool release];

}

void MacOpenGLWindow::runMainLoop()
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    // FILE* dump = fopen ("/Users/erwincoumans/yes.txt","wb");
    // fclose(dump);
    
   

    
    [pool release];

}

void MacOpenGLWindow::exit()
{
    
    delete m_internalData;
    m_internalData = 0;
    
}
extern float m_azi;
extern float m_ele;
extern float m_cameraDistance;


void MacOpenGLWindow::startRendering()
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    
    
    
    
    NSEvent *event = nil;
    
    do
    {
        [pool release];
        pool = [[NSAutoreleasePool alloc] init];
        event =
        [m_internalData->m_myApp
         nextEventMatchingMask:NSAnyEventMask
         untilDate:[NSDate distantPast]
         inMode:NSDefaultRunLoopMode
         //		  inMode:NSEventTrackingRunLoopMode
         dequeue:YES];
        
        if ([event type] == NSKeyDown)
        {
            uint32 keycode = [event keyCode];
            //printf("keycode = %d\n", keycode);
            if (keycode==49)
            {
                pauseSimulation = !pauseSimulation;
            }
         //   if (keycode==12)//'q'
           //     [NSApp terminate:m_internalData->m_myApp];
            
      //      if (keycode == 0)
        //        m_azi += 0.1;
        //    input::doSomeWork(keycode);
        }
         
        
        if ([event type]== NSRightMouseDown)
        {
           // printf("right mouse!");
            float mouseX,mouseY;
            
            NSPoint eventLocation = [event locationInWindow];
            NSPoint center = [m_internalData->m_myview convertPoint:eventLocation fromView:nil];
            m_mouseX = center.x;
            m_mouseY = center.y;
            shootObject = 1;
            
           // printf("mouse coord = %f, %f\n",mouseX,mouseY);
            
            
        }
        
        if ([event type] == NSMouseMoved)
        {
            NSPoint eventLocation = [event locationInWindow];
            NSPoint center = [m_internalData->m_myview convertPoint:eventLocation fromView:nil];
            m_mouseX = center.x;
            m_mouseY = center.y;
           // printf("mouse coord = %f, %f\n",m_mouseX,m_mouseY);
        }
        
        if ([event type] == NSLeftMouseDragged)
        {
            CGMouseDelta dx1, dy1;
            CGGetLastMouseDelta (&dx1, &dy1);
            
            //hack to avoid first time skip in delta mouse event
            static bool firstTime = true;
            if (!firstTime)
            {
          //      m_azi += dx1*0.1;
            //    m_ele += dy1*0.1;
            }
            firstTime = false;
            
            NSPoint eventLocation = [event locationInWindow];
            NSPoint center = [m_internalData->m_myview convertPoint:eventLocation fromView:nil];
            m_mouseX = center.x;
            m_mouseY = center.y;
            
            if (m_mouseCallback)
                (*m_mouseCallback)(1,0,m_mouseX,m_mouseY);

          //  printf("mouse coord = %f, %f\n",m_mouseX,m_mouseY);
        }
        
        if ([event type] == NSScrollWheel)
        {
            float dy, dx;
            dy = [ event deltaY ];
            dx = [ event deltaX ];
            if (m_wheelCallback)
                (*m_wheelCallback)(dx,dy);
          //  m_cameraDistance -= dy*0.1;
            // m_azi -= dx*0.1;
            
        }
        [m_internalData->m_myApp sendEvent:event];
        [m_internalData->m_myApp updateWindows];
    } while (event);
  
    [m_internalData->m_myview MakeCurrent];
    GLint err = glGetError();
    assert(err==GL_NO_ERROR);
    
    
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);     //clear buffers

    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    //glCullFace(GL_BACK);
    //glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    float aspect;
    //btVector3 extents;
    
    if (m_internalData->m_width > m_internalData->m_height)
    {
        aspect = (float)m_internalData->m_width / (float)m_internalData->m_height;
        //extents.setValue(aspect * 1.0f, 1.0f,0);
    } else
    {
        aspect = (float)m_internalData->m_height / (float)m_internalData->m_width;
        //extents.setValue(1.0f, aspect*1.f,0);
    }
    
    err = glGetError();
    assert(err==GL_NO_ERROR);
     [pool release];

}

void MacOpenGLWindow::endRendering()
{
    [m_internalData->m_myview MakeCurrent];
    glSwapAPPLE();
}

bool MacOpenGLWindow::requestedExit()
{
    return m_internalData->m_exitRequested;   
}

void MacOpenGLWindow::getMouseCoordinates(int& x, int& y)
{
    
    NSPoint pt = [m_internalData->m_window mouseLocationOutsideOfEventStream];
    m_mouseX = pt.x;
    m_mouseY = pt.y;
    
    x = m_mouseX;
    //our convention is x,y is upper left hand side
    y = [m_internalData->m_myview GetWindowHeight]-m_mouseY;

    
}




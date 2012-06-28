#include "MacOpenGLWindow.h"


#import <Cocoa/Cocoa.h>
#include <OpenGL/gl3.h>

#include <OpenGL/glext.h>

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>




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


void display(void)
{
	checkError("pre display");
    
	// clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
		
	checkError("display");
}

void reshape(int w, int h)
{
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
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

@end

float loop;

#define Pi 3.1415

@implementation TestView

-(void)drawRect:(NSRect)rect
{
	if (([self frame].size.width != m_lastWidth) || ([self frame].size.height != m_lastHeight))
	{
		m_lastWidth = [self frame].size.width;
		m_lastHeight = [self frame].size.height;
		
		// Only needed on resize:
		[m_context clearDrawable];
		
		reshape([self frame].size.width, [self frame].size.height);
	}
	
	[m_context setView: self];
	[m_context makeCurrentContext];
	
	// Draw
	display();
	
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
:m_internalData(0)
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
    NSString* appNameString = @"étape";
    NSString* menuItemString = @"mymenu";
    id appMenu = [[NSMenu new] autorelease];
    id appName = appNameString;//[[NSProcessInfo processInfo] processName];
    id quitTitle = [@"Quit " stringByAppendingString:appName];
    id quitMenuItem = [[[NSMenuItem alloc] initWithTitle:quitTitle
                                                  action:@selector(terminate:) keyEquivalent:@"q"] autorelease];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];
    
    
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

}

void MacOpenGLWindow::runMainLoop()
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    // FILE* dump = fopen ("/Users/erwincoumans/yes.txt","wb");
    // fclose(dump);
    
    [m_internalData->m_myApp finishLaunching];
#if 0 
    bool shouldKeepRunning = YES;
    do
    {
        [pool release];
        pool = [[NSAutoreleasePool alloc] init];
        printf(".");
        NSEvent *event =
        [m_internalData->m_myApp
         nextEventMatchingMask:NSAnyEventMask
         untilDate:[NSDate distantPast]
         inMode:NSDefaultRunLoopMode
         //		  inMode:NSEventTrackingRunLoopMode
         dequeue:YES];
        usleep(10000);
        //nanosleep(10000000);
        if ([event type] == NSKeyDown)
        {
            shouldKeepRunning=NO;
            //[NSApp terminate:self];
            //       printf("keydown\n");
        }        
        
        [m_internalData->m_myApp sendEvent:event];
        [m_internalData->m_myApp updateWindows];
    } while (shouldKeepRunning);
#endif
    
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
            if (keycode==12)//'q'
                [NSApp terminate:m_internalData->m_myApp];
            
            if (keycode == 0)
                m_azi += 0.1;
        //    input::doSomeWork(keycode);
        }
        if ([event type] == NSLeftMouseDragged)
        {
            CGMouseDelta dx1, dy1;
            CGGetLastMouseDelta (&dx1, &dy1);
            m_azi += dx1*0.1;
            m_ele += dy1*0.1;
        }
        
        if ([event type] == NSScrollWheel)
        {
            float dy, dx;
            dy = [ event deltaY ];
            dx = [ event deltaX ];
            m_cameraDistance -= dy*0.1;
            
        }
        [m_internalData->m_myApp sendEvent:event];
        [m_internalData->m_myApp updateWindows];
    } while (event);
  
    [m_internalData->m_myview MakeCurrent];
    
    
    //glClearColor(1.f,0.f,0.f,1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);     //clear buffers

    GLint err = glGetError();
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
    
}


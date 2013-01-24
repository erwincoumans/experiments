/*
 Oolong Engine for the iPhone / iPod touch
 Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/
 
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose, 
 including commercial applications, and to alter it and redistribute it freely, 
 subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#import "App.h"
#import "Delegate.h"
#include "Camera.h"

//CONSTANTS:
#define kFPS			60.0
#define kSpeed			10.0

static CShell *shell = NULL;

@implementation AppController

- (void) update
{
	if(!shell->UpdateScene())
		printf("UpdateScene error\n");
	
    if(!shell->RenderScene())
		printf("RenderScene error\n");
	
	[_glView swapBuffers];
}


- (void) applicationDidFinishLaunching:(UIApplication*)application
{
	CGRect	rect = [[UIScreen mainScreen] bounds];
	
	// create a full-screen window
	_window = [[UIWindow alloc] initWithFrame:rect];
	
	// create the OpenGL view and add it to the window
	//_glView = [[EAGLView alloc] initWithFrame:rect];
	_glView = [[EAGLView alloc] initWithFrame:rect pixelFormat:GL_RGB565_OES depthFormat:GL_DEPTH_COMPONENT16_OES preserveBackbuffer:NO];
	
	[_window addSubview:_glView];

	// show the window
	[_window makeKeyAndVisible];
	
	if(!shell->InitApplication())
		printf("InitApplication error\n");
	
	// create our rendering timer
	[NSTimer scheduledTimerWithTimeInterval:(1.0 / kFPS) target:self selector:@selector(update) userInfo:nil repeats:YES];
}

- (void) dealloc
{
	if(!shell->QuitApplication())
		printf("QuitApplication error\n");

	[_glView release];
	[_window release];
	
	[super dealloc];
}

@end

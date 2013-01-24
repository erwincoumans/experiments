
#ifndef APPLICATION_H_
#define APPLICATION_H_

//
//
/// \brief Interface class for an oolong engine based app. 
/// 
/// New apps should implement this class' member functions to define their app's high level behaviour.
class CShell
{
public:
	//! \brief initialization before the render API is intialized
	bool InitApplication();
	
	//! \brief release any memory/resources acquired by InitApplication()
	bool QuitApplication();
/*
	//! \brief called to initialise the view.
        //!
        //! It is called any time the rendering API is initialised,
	//! i.e. once at the beginning, and possibly again if the
	//! resolution changes, or a power management even occurs, or
	//! if the app requests a reinialisation.
	//! The application should check here the configuration of
	//! the rendering API; it is possible that requests made in
	//! InitApplication() were not successful.
	//! Since everything is initialised when this function is
	//! called, you can load textures and perform rendering API
	//! functions.
	bool InitView();

	//! \brief Called as the view is released
        //!
        //! It will be called after the RenderScene() loop, before
	//! shutting down the render API. It enables the application
	//! to release any memory/resources acquired in InitView().
	bool ReleaseView();
*/
	//! \brief update the camera matrix and other things that need to be done to setup rendering
	bool UpdateScene();

	//! \brief It is main application function in which you have to do your own rendering.  Will be
	//! called repeatedly until the application exits.
	bool RenderScene();
};


#endif APPLICATION_H_



/**
 * @mainpage Oolong Engine Documentation
 *
 * @section intro_sec Introduction
 * Oolong Engine
 *
 * The Oolong Engine is written in C++ with some help from Objective-C. It will help you to create new games and port existing games to the iPhone and the iPod touch. Here is its feature list:
 *      - OpenGL ES 1.1 support
 *      - Math library that supports fixed-point and floating-point calculations with an interface very similar to the D3D math library
 *      - Support for numerous texture formats including the PowerVR 2-bit, 4-bit and normal map compression formats
 *      - Support for PowerVR's POD (Scene and Meshes) and the 3DS files formats
 *      - Touch screen support
 *      - Accelerometer support
 *      - Text rendering to support a basic UI
 *      - Timing: several functions that can replace rdstc, QueryPerformance? etc.
 *      - Profiler: industry proven in-game profiler
 *      - Resources streaming system
 *      - Bullet SDK support (for 3D Physics)
 *      - Audio engine with OpenAL support 
 * 
 * The Library is Open Source and free for commercial use, under the ZLib license ( http://opensource.org/licenses/zlib-license.php )
 *
 * @section install_sec Installation
 * @subsection step1 Step 1: Download
 * You can download the Oolong engine from our website: http://oolongengine.com/
 *
 * You will need the apple iphone SDK, available from: http://developer.apple.com/iphone/
 * @subsection step2 Step 2: Building
 * Oolong engine comes with a set of demo apps under the Examples subdirectory, you can load and build these examples by locating the example's .xcodeproj file and loading it within XCode. From there simply select build and go to build and run the example application with the iphone SDK's simulator.
 * @subsection step3 Step 3: Creating your own application
 * TODO
 * @section copyright Copyright
 * Copyright (c) xxxx-xxxx Wolfgang Engel
 *
 * Oolong engine uses the Bullet Collision Detection & Physics SDK. The Bullet Library is Open Source and free for commercial use, under the ZLib license ( http://opensource.org/licenses/zlib-license.php ).
 *
 * Bullet SDK: Copyright (C) 2005-2007 Erwin Coumans, some contributions Copyright Gino van den Bergen, Christer Ericson, Simon Hobbs, Ricardo Padrela, F Richter(res), Stephane Redon
 * Special thanks to all visitors of the Bullet Physics forum, and in particular above contributors, Dave Eberle, Dirk Gregorius, Erin Catto, Dave Eberle, Adam Moravanszky,
 * Pierre Terdiman, Kenny Erleben, Russell Smith, Oliver Strunk, Jan Paul van Waveren, Marten Svanfeldt.
 */

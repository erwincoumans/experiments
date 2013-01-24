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
#include "Timer.h"

int GetFps(int frame, CFTimeInterval &TimeInterval, CFTimeInterval *pFrameTime )
{
	// do all the timing
	static CFTimeInterval startTime = 0;
	static CFTimeInterval lastFrameTime = 0;
	
	int frameRate = 0;

	// calculate our local time
	TimeInterval = CFAbsoluteTimeGetCurrent();
	
	if(startTime == 0)
		startTime = TimeInterval;
	
	if(lastFrameTime == 0)
		lastFrameTime = TimeInterval;
	if( pFrameTime ) {
		*pFrameTime = TimeInterval - lastFrameTime;
	}
	lastFrameTime = TimeInterval;

	TimeInterval = TimeInterval - startTime;

	if (TimeInterval) 
		frameRate = ((float)frame/(TimeInterval)) + 1.0f;

	return frameRate;
}

void StartTimer(structTimer* timer)
{
	/* Get the timebase info */
	mach_timebase_info(&timer->info);
}

void ResetTimer(structTimer* timer)
{
	// calculate our local time
	timer->startTime = mach_absolute_time();
}

float GetAverageTimeValueInMS(structTimer* timer)
{
	uint64_t TimeInterval;
	
	// calculate our local time
	TimeInterval = mach_absolute_time();
	
	U32 duration = (U32)(TimeInterval - timer->startTime);
	
	/* Convert to nanoseconds */
	duration *= timer->info.numer;
	duration /= timer->info.denom;

	// return in milliseconds
	return ((float)duration) / 1000.0f;	
}


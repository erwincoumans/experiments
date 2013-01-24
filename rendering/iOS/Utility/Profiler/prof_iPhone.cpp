
//#include "Timer.h"
//#include <mach/mach.h>
//#include <mach/mach_time.h>

//#include <Mathematics.h>
//#include <Log.h>
#include <Timing.h>


double Prof_get_time(void)
{
	return (double) GetTimeInNsSinceCPUStart();
	//return (double) GetTimeInMsSince1970();
/*
   BOOL ok = QueryPerformanceFrequency(&freq);
   assert(ok == TRUE);

   freq.QuadPart = freq.QuadPart;

   ok = QueryPerformanceCounter(&time);
   assert(ok == TRUE);

   return time.QuadPart / (double) freq.QuadPart;
*/
}

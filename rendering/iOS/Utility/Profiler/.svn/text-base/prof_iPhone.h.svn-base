#ifndef Prof_INC_PROF_IPHONE_H
#define Prof_INC_PROF_IPHONE_H

#include <Mathematics.h>
//#include <Log.h>
#include <Timing.h>

typedef INT64BIT Prof_Int64;
#include <mach/mach.h>
#include <mach/mach_time.h>

#ifdef __cplusplus
  inline
#elif _MSC_VER >= 1200
  __forceinline
#else
  static
#endif

	// time when a certain event occurs
	// rdtsc returns a 64-bit value in registers EDX:EAX that represents the count of ticks from processor reset.
	// the "time stamp" counter. This is a register internal to the 
    // CPU, which counts, using a 64 bit timer, the number of CPU 
    // clocks executed since system power-on (more or less).
      void Prof_get_timestamp(Prof_Int64 *result)
      {
		INT64BIT time = (INT64BIT)GetTimeInNsSinceCPUStart();
		//INT64BIT time = (INT64BIT)GetTimeInTicksSinceCPUStart();
		
		
		result = &time;
	  /*
         __asm {
            rdtsc;
            mov    ebx, result
            mov    [ebx], eax
            mov    [ebx+4], edx
         }
		 */
      }

#endif

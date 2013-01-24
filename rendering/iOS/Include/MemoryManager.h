#ifndef MEMORY_H_
#define MEMORY_H_

#ifdef __APPLE__
#include <TargetConditionals.h>

#include "../Utility/MemoryManager/MemoryTemplates.h"

#if((TARGET_IPHONE_SIMULATOR == 0) && (TARGET_OS_IPHONE == 1) && (USEMEMORYMANAGER))
#include "../Utility/MemoryManager/mmgr.h"
#else
#include "../Utility/MemoryManager/nommgr.h"
#endif



#endif
#endif

#ifndef BTB_BASE_H
#define BTB_BASE_H


#define BTB_DECLARE_HANDLE(name) typedef struct name##__ { int unused; } *name

#ifdef _DEBUG
	#include <assert.h>
	#define btbAssert(a) assert(a);
#else
	#define btbAssert(a)
#endif

BTB_DECLARE_HANDLE(btbDevice);
BTB_DECLARE_HANDLE(btbBuffer);

typedef struct
{
	unsigned int m_key;
} btbSortData1ui;

typedef struct
{
	unsigned int m_key;
	unsigned int m_value;
} btbSortData2ui;

BTB_DECLARE_HANDLE(btbRadixSort);



#endif //BTB_BASE_H

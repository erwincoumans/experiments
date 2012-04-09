#ifndef BTB_BASE_H
#define BTB_BASE_H

#ifdef __cplusplus
extern "C" {
#endif//__cplusplus

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

enum btbScalarType
{
	BTB_SIGNED_INT8=12,
	BTB_SIGNED_INT16,
	BTB_SIGNED_INT32,
	BTB_FLOAT_TYPE,
};

BTB_DECLARE_HANDLE(btbRadixSort);


#ifdef __cplusplus
}
#endif//__cplusplus


#endif //BTB_BASE_H

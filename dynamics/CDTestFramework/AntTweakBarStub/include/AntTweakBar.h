#ifndef ANT_TWEAK_BAR_GWEN_WRAPPER_H
#define ANT_TWEAK_BAR_GWEN_WRAPPER_H

#define TW_CALL

struct	TwBar
{
	
};

enum TwTypes
{
TW_TYPE_FLOAT = 1,
TW_TYPE_BOOLCPP,
TW_TYPE_INT32
};

enum TwGraphAPI
{
	TW_OPENGL	=1,
	TW_DIRECT3D_9
};

typedef struct CTwEnumVal
{
    int           Value;
    const char *  Label;
} TwEnumVal;


typedef int  TwType;


int TwEventKeyboardGLUT(unsigned char glutKey, int mouseX, int mouseY);
int TwEventSpecialGLUT(int glutKey, int mouseX, int mouseY);
int TwEventMouseButtonGLUT(int glutButton, int glutState, int mouseX, int mouseY);
int TwEventMouseMotionGLUT(int mouseX, int mouseY);
int       TwDraw();
int      TwWindowSize(int width, int height);
int          TwDeleteBar(TwBar *bar);

int      TwInit(TwGraphAPI graphAPI, void *device);
int      TwTerminate();
const char * TwGetLastError();
typedef void (*GLUTmousemotionfun)(int mouseX, int mouseY);
int TwGLUTModifiersFunc(int (*glutGetModifiersFunc)(void));
TwBar *      TwNewBar(const char *barName);
TwType   TwDefineEnum(const char *name, const TwEnumVal *enumValues, unsigned int nbValues);
int      TwAddVarRW(TwBar *bar, const char *name, TwType type, void *var, const char *def);
void TwAddVarRO(TwBar* tbar, const char* , TwType, void* ,const char* );
typedef void (* TwButtonCallback)(void *clientData);

int      TwAddButton(TwBar *bar, const char *name, TwButtonCallback callback, void *clientData, const char *def);


#endif //ANT_TWEAK_BAR_GWEN_WRAPPER_H
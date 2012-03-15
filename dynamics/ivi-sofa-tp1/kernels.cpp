#include "kernels.h"
#include <string>
#include <sstream>

#ifdef SOFA_DEVICE_CPU
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

//// DATA ////
int cuda_device = -1;
std::string device_name;

//// METHODS ////

#ifdef SOFA_DEVICE_CPU
static void cpuid(unsigned int a, unsigned int b[4])
{
#if defined(_MSC_VER)
    __cpuid((int*)b,a);
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    asm volatile("xchgl %%ebx, %1\n"
                 "cpuid\n"
                 "xchgl %%ebx, %1\n"
                 :"=a"(*b),"=r"(*(b+1)),
                 "=c"(*(b+2)),"=d"(*(b+3)):"0"(a));
#else
    b[0] = b[1] = b[2] = b[3] = 0;
#endif
}

std::string cpu_name()
{
    unsigned int b[13] = {0};
    cpuid(0x80000000,b);
    unsigned int max = b[0];
    if (max < 0x80000004) return std::string();
    cpuid(0x80000002,b);
    cpuid(0x80000003,b+4);
    cpuid(0x80000004,b+8);
    std::string s;
    b[12] = 0;
    const char* p = (const char*)b;
    char last = '\0';
    while (*p)
    {
        char c = *p; ++p;
        if (c == ' ' && last == ' ') continue;
        if (c == '(')
        {
            while (*p && c != ')') c = *p++;
            continue;
        }
        s += c; last = c;
    }
    return s;
}
#endif

bool kernels_init()
{
    std::ostringstream o;
#if defined(SOFA_DEVICE_CUDA)
    o << "GPU:";
    if (!mycudaInit(cuda_device))
        return false;
    o << " " << mycudaGetDeviceName();
#elif defined(SOFA_DEVICE_CPU)
    o << "CPU:";
    o << " " << cpu_name();
#endif
    device_name = o.str();
#if defined(VERSION)
    std::cout << "V" << VERSION << " ";
#endif
    std::cout << device_name << " (" << sizeof(void*)*8 << " bits)" << std::endl;
    return true;
}

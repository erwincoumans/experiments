/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_MEMORYMANAGER_H
#define SOFA_HELPER_MEMORYMANAGER_H

#include <sofa/helper/helper.h>
#include <cstring>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace helper
{

#ifndef MAX_NUMBER_OF_DEVICES
#define MAX_NUMBER_OF_DEVICES 8
#endif

/* Generic MemoryManager
 * Its use is as a template parameter only and it cannot be instanciated (link error otherwise).
 */
template <class T>
class MemoryManager
{
public:
    typedef T* host_pointer;
    typedef const T* const_host_pointer;

    //have to be changed according to the type of device
    typedef void* device_pointer;
    typedef const void* const_device_pointer;

    typedef GLuint gl_buffer;

    enum { MAX_DEVICES = 0 };
    enum { BSIZE = 1 };
    enum { SUPPORT_GL_BUFFER = 0 };

    static void hostAlloc(host_pointer* hPointer,int n) { *hPointer = new T[n/sizeof(T)]; }
    static void memsetHost(host_pointer hPointer, int value,size_t n) { memset((void*) hPointer, value, n); }
    static void hostFree(const_host_pointer hSrcPointer) { delete[] hSrcPointer; }
    
    static int numDevices();

    //static void deviceAlloc(int d,device_pointer* dPointer, int n);
    //static void deviceFree(int d,const_device_pointer dSrcPointer);
    //static void memcpyHostToDevice(int d, device_pointer dDestPointer, const_host_pointer hSrcPointer, size_t n);
    //static void memcpyDeviceToHost(int d, host_pointer hDestPointer, const_void * dSrcPointer , size_t n);
    //static void memcpyDeviceToDevice(int d, device_pointer dDestPointer, const_device_pointer dSrcPointer , size_t n);
    //static void memsetDevice(int d, device_pointer dDestPointer, int value,size_t n);
    
    static int getBufferDevice() { return 0; }
    
    static bool bufferAlloc(gl_buffer* bId, int n) { return false; }
    static void bufferFree(const gl_buffer bId) {}
    
    static bool bufferRegister(const gl_buffer bId) { return false; }
    static void bufferUnregister(const gl_buffer bId) {}
    static bool bufferMapToDevice(device_pointer* dDestPointer, const gl_buffer bSrcId) { return false; }
    static void bufferUnmapToDevice(device_pointer* dDestPointer, const gl_buffer bSrcId) {}

    //static device_pointer deviceOffset(device_pointer dPointer,size_t offset){return (T*)dPointer+offset;}
    //static device_pointer null(){return NULL;}
    //static bool isNull(device_pointer p){return p==NULL;}
};

//CPU MemoryManager
template <class T >
class CPUMemoryManager : public MemoryManager<T>
{
public:
    typedef T* host_pointer;
    typedef const T* const_host_pointer;
    // on CPU there is not device, so we define the device_pointer type to be the same as the host_pointer
    typedef host_pointer device_pointer;
    typedef const_host_pointer const_device_pointer;

    static device_pointer deviceOffset(device_pointer dPointer,size_t offset) { return (T*)dPointer+offset; }
    static const_device_pointer deviceOffset(const_device_pointer dPointer,size_t offset) { return (T*)dPointer+offset; }
    static device_pointer null() { return NULL; }
    static bool isNull(const_device_pointer p) { return p==NULL; }
};

}

}

#endif //SOFA_HELPER_MEMORYMANAGER_H



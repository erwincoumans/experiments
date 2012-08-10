/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#ifndef GL_INSTANCING_RENDERER_H
#define GL_INSTANCING_RENDERER_H

#include "LinearMath/btAlignedObjectArray.h"

void btDefaultMouseCallback( int button, int state, float x, float y);
void btDefaultKeyboardCallback(unsigned char key, int x, int y);

class GLInstancingRenderer
{
	
	btAlignedObjectArray<struct btGraphicsInstance*> m_graphicsInstances;

	int		m_maxObjectCapacity;
	int		m_maxShapeCapacity;
	struct InternalDataRenderer* m_data;

	
public:
	GLInstancingRenderer(int m_maxObjectCapacity, int maxShapeCapacity = 512*1024);
	virtual ~GLInstancingRenderer();

	void InitShaders();
	void RenderScene(void);
	void CleanupShaders();

	///vertices must be in the format x,y,z, nx,ny,nz, u,v
	int registerShape(const float* vertices, int numvertices, const int* indices, int numIndices);

	///position x,y,z, quaternion x,y,z,w, color r,g,b,a, scaling x,y,z
	int registerGraphicsInstance(int shapeIndex, const float* position, const float* quaternion, const float* color, const float* scaling);

	void writeTransforms();

	void writeSingleInstanceTransformToCPU(float* position, float* orientation, int srcIndex);

	void writeSingleInstanceTransformToGPU(float* position, float* orientation, int srcIndex);

	void writeSingleInstanceColorToCPU(float* color, int srcIndex);

	void getMouseDirection(float* dir, int mouseX, int mouseY);

	void updateCamera();

	void	getCameraPosition(float cameraPos[4]);
	void	setCameraDistance(float dist);
	float	getCameraDistance() const;

	int getMaxShapeCapacity() const
	{
		return m_maxShapeCapacity;
	}
};

#endif //GL_INSTANCING_RENDERER_H

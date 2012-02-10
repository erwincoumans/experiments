#ifndef GL_INSTANCING_RENDERER_H
#define GL_INSTANCING_RENDERER_H

#include "LinearMath/btAlignedObjectArray.h"

class GLInstancingRenderer
{
	
	btAlignedObjectArray<struct btGraphicsInstance*> m_graphicsInstances;

	struct InternalDataRenderer* m_data;

public:
	GLInstancingRenderer();
	virtual ~GLInstancingRenderer();

	void InitShaders();
	void RenderScene(void);
	void CleanupShaders();

	///vertices must be in the format x,y,z, nx,ny,nz, u,v
	int registerShape(const float* vertices, int numvertices, const int* indices, int numIndices);

	///position x,y,z, quaternion x,y,z,w, color r,g,b,a, scaling x,y,z
	int registerGraphicsInstance(int shapeIndex, const float* position, const float* quaternion, const float* color, const float* scaling);

	void writeTransforms();
};

#endif //GL_INSTANCING_RENDERER_H

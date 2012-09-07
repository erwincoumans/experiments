#ifndef GLES2_SHAPE_DRAWER_H
#define GLES2_SHAPE_DRAWER_H

#include "LinearMath/btVector3.h"

class btCollisionShape;

class GLES2ShapeDrawer
{
public:
	void drawOpenGL(int modelMatrix, int positionLoc, int texCoordLoc, int localScaling,btScalar* m, const btCollisionShape* shape, const btVector3& color,int	debugMode,const btVector3& worldBoundsMin,const btVector3& worldBoundsMax);

};

#endif//GLES2_SHAPE_DRAWER_H

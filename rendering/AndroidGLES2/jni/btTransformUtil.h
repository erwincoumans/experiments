#ifndef _BT_TRANSFORM_UTIL_GL_H
#define _BT_TRANSFORM_UTIL_GL_H

#include "LinearMath/btVector3.h"

void	btCreateFrustum(		float left, 		float right, 		float bottom, 		float top, 
		float nearVal, 		float farVal,		float frustum[16]);
		
void	btCreateLookAt(const btVector3& eye, const btVector3& center,const btVector3& up, float result[16]);

#endif //_BT_TRANSFORM_UTIL_GL_H

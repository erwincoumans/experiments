

#include "btTransformUtil.h"

void	btCreateFrustum(
		float left, 
		float right, 
		float bottom, 
		float top, 
		float nearVal, 
		float farVal,
		float frustum[16])
{
	
		frustum[0*4+0] = (float(2) * nearVal) / (right - left);
		frustum[0*4+1] = float(0);
		frustum[0*4+2] = float(0);
		frustum[0*4+3] = float(0);

		frustum[1*4+0] = float(0);
		frustum[1*4+1] = (float(2) * nearVal) / (top - bottom);
		frustum[1*4+2] = float(0);
		frustum[1*4+3] = float(0);

		frustum[2*4+0] = (right + left) / (right - left);
		frustum[2*4+1] = (top + bottom) / (top - bottom);
		frustum[2*4+2] = -(farVal + nearVal) / (farVal - nearVal);
		frustum[2*4+3] = float(-1);

		frustum[3*4+0] = float(0);
		frustum[3*4+1] = float(0);
		frustum[3*4+2] = -(float(2) * farVal * nearVal) / (farVal - nearVal);
		frustum[3*4+3] = float(0);

}
#include <string.h>

void	btCreateLookAt(const btVector3& eye, const btVector3& center,const btVector3& up, float result[16])
{
        btVector3 f = (center - eye).normalized();
        btVector3 u = up.normalized();
		btVector3 s = (f.cross(u)).normalized();
        u = s.cross(f);

		result[0*4+0] = s.x();
        result[1*4+0] = s.y();
        result[2*4+0] = s.z();
		result[3*4+0] = -s.dot(eye);

		result[0*4+1] = u.x();
		result[1*4+1] = u.y();
        result[2*4+1] = u.z();
		result[3*4+1] = -u.dot(eye);

		result[0*4+2] =-f.x();
        result[1*4+2] =-f.y();
		result[2*4+2] =-f.z();
		result[3*4+2] = f.dot(eye);

		result[0*4+3] = 0.f;
		result[1*4+3] = 0.f;
		result[2*4+3] = 0.f;
		result[3*4+3] = 1.f;
}

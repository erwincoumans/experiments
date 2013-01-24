/******************************************************************************

 @File         PVRTVector.cpp

 @Title        

 @Copyright    Copyright (C) 2007 - 2008 by Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Vector and matrix mathematics library

******************************************************************************/
#include "Vector.h"

#include <math.h>

/*!***************************************************************************
** Vec2 2 component vector
****************************************************************************/

/*!***************************************************************************
 @Function			Vec2
 @Input				v3Vec a Vec3
 @Description		Constructor from a Vec3
 *****************************************************************************/
/*	Vec2::Vec2(const Vec3& vec3)
	{
		x = vec3.x; y = vec3.y;
	}
*/
/*!***************************************************************************
** Vec3 3 component vector
****************************************************************************/

/*!***************************************************************************
 @Function			Vec3
 @Input				v4Vec a Vec4
 @Description		Constructor from a Vec4
*****************************************************************************/
/*	Vec3::Vec3(const Vec4& vec4)
	{
		x = vec4.x; y = vec4.y; z = vec4.z;
	}
*/
/*!***************************************************************************
 @Function			*
 @Input				rhs a PVRTMat3
 @Returns			result of multiplication
 @Description		matrix multiplication operator Vec3 and PVRTMat3
****************************************************************************/
/*	Vec3 Vec3::operator*(const PVRTMat3& rhs) const
	{
		Vec3 out;

		out.x = VERTTYPEMUL(x,rhs.f[0])+VERTTYPEMUL(y,rhs.f[1])+VERTTYPEMUL(z,rhs.f[2]);
		out.y = VERTTYPEMUL(x,rhs.f[3])+VERTTYPEMUL(y,rhs.f[4])+VERTTYPEMUL(z,rhs.f[5]);
		out.z = VERTTYPEMUL(x,rhs.f[6])+VERTTYPEMUL(y,rhs.f[7])+VERTTYPEMUL(z,rhs.f[8]);

		return out;
	}
*/
/*!***************************************************************************
 @Function			*=
 @Input				rhs a PVRTMat3
 @Returns			result of multiplication and assignment
 @Description		matrix multiplication and assignment operator for Vec3 and PVRTMat3
****************************************************************************/
/*	Vec3& Vec3::operator*=(const PVRTMat3& rhs)
	{
		VERTTYPE tx = VERTTYPEMUL(x,rhs.f[0])+VERTTYPEMUL(y,rhs.f[1])+VERTTYPEMUL(z,rhs.f[2]);
		VERTTYPE ty = VERTTYPEMUL(x,rhs.f[3])+VERTTYPEMUL(y,rhs.f[4])+VERTTYPEMUL(z,rhs.f[5]);
		z = VERTTYPEMUL(x,rhs.f[6])+VERTTYPEMUL(y,rhs.f[7])+VERTTYPEMUL(z,rhs.f[8]);
		x = tx;
		y = ty;

		return *this;
	}
*/
/*!***************************************************************************
** Vec4 4 component vector
****************************************************************************/

/*!***************************************************************************
 @Function			*
 @Input				rhs a PVRTMat4
 @Returns			result of multiplication
 @Description		matrix multiplication operator Vec4 and PVRTMat4
****************************************************************************/
/*	Vec4 Vec4::operator*(const PVRTMat4& rhs) const
	{
		Vec4 out;
		out.x = VERTTYPEMUL(x,rhs.f[0])+VERTTYPEMUL(y,rhs.f[1])+VERTTYPEMUL(z,rhs.f[2])+VERTTYPEMUL(w,rhs.f[3]);
		out.y = VERTTYPEMUL(x,rhs.f[4])+VERTTYPEMUL(y,rhs.f[5])+VERTTYPEMUL(z,rhs.f[6])+VERTTYPEMUL(w,rhs.f[7]);
		out.z = VERTTYPEMUL(x,rhs.f[8])+VERTTYPEMUL(y,rhs.f[9])+VERTTYPEMUL(z,rhs.f[10])+VERTTYPEMUL(w,rhs.f[11]);
		out.w = VERTTYPEMUL(x,rhs.f[12])+VERTTYPEMUL(y,rhs.f[13])+VERTTYPEMUL(z,rhs.f[14])+VERTTYPEMUL(w,rhs.f[15]);
		return out;
	}
*/
/*!***************************************************************************
 @Function			*=
 @Input				rhs a PVRTMat4
 @Returns			result of multiplication and assignment
 @Description		matrix multiplication and assignment operator for Vec4 and PVRTMat4
****************************************************************************/
/*	Vec4& Vec4::operator*=(const PVRTMat4& rhs)
	{
		VERTTYPE tx = VERTTYPEMUL(x,rhs.f[0])+VERTTYPEMUL(y,rhs.f[1])+VERTTYPEMUL(z,rhs.f[2])+VERTTYPEMUL(w,rhs.f[3]);
		VERTTYPE ty = VERTTYPEMUL(x,rhs.f[4])+VERTTYPEMUL(y,rhs.f[5])+VERTTYPEMUL(z,rhs.f[6])+VERTTYPEMUL(w,rhs.f[7]);
		VERTTYPE tz = VERTTYPEMUL(x,rhs.f[8])+VERTTYPEMUL(y,rhs.f[9])+VERTTYPEMUL(z,rhs.f[10])+VERTTYPEMUL(w,rhs.f[11]);
		w = VERTTYPEMUL(x,rhs.f[12])+VERTTYPEMUL(y,rhs.f[13])+VERTTYPEMUL(z,rhs.f[14])+VERTTYPEMUL(w,rhs.f[15]);
		x = tx;
		y = ty;
		z = tz;
		return *this;
	}
*/

/*****************************************************************************
End of file (Vector.cpp)
*****************************************************************************/

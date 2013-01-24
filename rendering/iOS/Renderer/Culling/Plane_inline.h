/*
 Oolong Engine for the iPhone / iPod touch
 Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/
 
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose, 
 including commercial applications, and to alter it and redistribute it freely, 
 subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */
#ifndef PLANE_INLINE_H_
#define PLANE_INLINE_H_

#include "Mathematics.h"

#include "Plane.h"
/* Probably wrong and not ready for fixed-point arithemtic
inline void CPlane::ComputePlane(const VECTOR3& a, const VECTOR3& b, const VECTOR3 &c)
{
	VECTOR3 v1, v2, n, NormalizedN;
	
	v1.x = a.x - b.x;
	v1.y = a.y - b.y;
	v1.z = a.z - b.z;
	
	v2.x = c.x - b.x;
	v2.y = c.y - b.y;
	v2.z = c.z - b.z;
	
	MatrixVec3CrossProduct(n, v1, v2);
	MatrixVec3Normalize(NormalizedN, n);
	
	Plane.x = NormalizedN.x;
	Plane.y = NormalizedN.y;
	Plane.z = NormalizedN.z;
	
	// find the distance from the plane to the origin
	// Plane.w = -NormalizedN.x * b.x - NormalizedN.y * b.y - NormalizedN.z * b.z;
	Plane.w = -(NormalizedN.x * b.x + NormalizedN.y * b.y + NormalizedN.z * b.z);
}

inline void CPlane::ComputePlaneFromPointNormal(const VECTOR3& point, const VECTOR3& normal)
{
	Plane.x = normal.x;
	Plane.y = normal.y;
	Plane.z = normal.z;
	//??? is that right ???
	// Plane.w = -normal.x * point.x - normal.y * point.y - normal.z * point.z;
	Plane.w = -(normal.x * point.x + normal.y * point.y + normal.z * point.z);

}
*/

inline VERTTYPE CPlane::DistanceToPlane(const VECTOR3& point)
{
	return MatrixVec3DotProduct((VECTOR3&)Plane, point) + Plane.w;
}

inline void CPlane::NormalizePlane()
{
/*
    VERTTYPE denom;
	
	CPlane * pP = Plane;

    denom = SQRT(pP->x * pP->x + pP->y * pP->y + pP->z * pP->z);
    //if (denom < MX_EPS5) return NULL;

    denom = 1.0f / denom;

    Plane.x = Plane.x * denom;
    Plane.y = Plane.y * denom;
    Plane.z = Plane.z * denom;
    Plane.w = Plane.w * denom;
*/
	MatrixVec4Normalize(Plane, Plane);
}


#endif // AABB_H_
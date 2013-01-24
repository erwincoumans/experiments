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
#include <math.h>
#include <string.h>

#include "Mathematics.h"

#include "Frustum.h"

inline void CFrustum::ExtractPlanes(const MATRIX& comboMatrix, bool normalize) 
{ 
	VECTOR4 TempPlane;
	
    // Left clipping plane
    TempPlane.x = comboMatrix.f[_41] + comboMatrix.f[_11]; 
    TempPlane.y = comboMatrix.f[_42] + comboMatrix.f[_12]; 
    TempPlane.z = comboMatrix.f[_43] + comboMatrix.f[_13]; 
    TempPlane.w = comboMatrix.f[_44] + comboMatrix.f[_14]; 
	Plane[0].Set(TempPlane);
	
    // Right clipping plane 
    TempPlane.x = comboMatrix.f[_41] - comboMatrix.f[_11]; 
    TempPlane.y = comboMatrix.f[_42] - comboMatrix.f[_12]; 
    TempPlane.z = comboMatrix.f[_43] - comboMatrix.f[_13]; 
    TempPlane.w = comboMatrix.f[_44] - comboMatrix.f[_14]; 
	Plane[1].Set(TempPlane);

    // Top clipping plane 
    TempPlane.x = comboMatrix.f[_41] - comboMatrix.f[_21]; 
    TempPlane.y = comboMatrix.f[_42] - comboMatrix.f[_22]; 
    TempPlane.z = comboMatrix.f[_43] - comboMatrix.f[_23]; 
    TempPlane.w = comboMatrix.f[_44] - comboMatrix.f[_24]; 
	Plane[2].Set(TempPlane);
	
    // Bottom clipping plane 
    TempPlane.x = comboMatrix.f[_41] + comboMatrix.f[_21]; 
    TempPlane.y = comboMatrix.f[_42] + comboMatrix.f[_22]; 
    TempPlane.z = comboMatrix.f[_43] + comboMatrix.f[_23]; 
    TempPlane.w = comboMatrix.f[_44] + comboMatrix.f[_24]; 
	Plane[3].Set(TempPlane);

    // Near clipping plane 
    TempPlane.x = comboMatrix.f[_41] + comboMatrix.f[_31]; 
    TempPlane.y = comboMatrix.f[_42] + comboMatrix.f[_32]; 
    TempPlane.z = comboMatrix.f[_43] + comboMatrix.f[_33]; 
    TempPlane.w = comboMatrix.f[_44] + comboMatrix.f[_34]; 
	Plane[4].Set(TempPlane);

    // Far clipping plane 
    TempPlane.x = comboMatrix.f[_41] - comboMatrix.f[_31]; 
    TempPlane.y = comboMatrix.f[_42] - comboMatrix.f[_32]; 
    TempPlane.z = comboMatrix.f[_43] - comboMatrix.f[_33]; 
    TempPlane.w = comboMatrix.f[_44] - comboMatrix.f[_34]; 
	Plane[5].Set(TempPlane);
	
    // Normalize the plane equations, if requested 
    if (normalize == true) 
    { 
        Plane[0].NormalizePlane(); 
        Plane[1].NormalizePlane(); 
        Plane[2].NormalizePlane(); 
        Plane[3].NormalizePlane(); 
        Plane[4].NormalizePlane(); 
        Plane[5].NormalizePlane(); 
    } 
} 

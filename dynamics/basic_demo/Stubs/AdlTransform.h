#ifndef _ADL_TRANSFORM_H
#define _ADL_TRANSFORM_H

#include "AdlMath.h"
#include "AdlQuaternion.h"
#include "AdlMatrix3x3.h"

struct Transform
{
	float4 m_translation;
	Matrix3x3 m_rotation;
};

Transform trSetTransform(const float4& translation, const Quaternion& quat)
{
	Transform tr;
	tr.m_translation = translation;
	tr.m_rotation = qtGetRotationMatrix( quat );
	return tr;
}

Transform trInvert( const Transform& tr )
{
	Transform ans;
	ans.m_rotation = mtTranspose( tr.m_rotation );
	ans.m_translation = mtMul1( ans.m_rotation, -tr.m_translation );
	return ans;
}

Transform trMul(const Transform& trA, const Transform& trB)
{
	Transform ans; 
	ans.m_rotation = mtMul( trA.m_rotation, trB.m_rotation );
	ans.m_translation = mtMul1( trA.m_rotation, trB.m_translation ) + trA.m_translation;
	return ans;
}

float4 trMul1(const Transform& tr, const float4& p)
{
	return mtMul1( tr.m_rotation, p ) + tr.m_translation;
}


#endif //_ADL_TRANSFORM_H


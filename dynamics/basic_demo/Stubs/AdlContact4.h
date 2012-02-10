#ifndef ADL_CONTACT4_H
#define ADL_CONTACT4_H

#ifdef CL_PLATFORM_AMD
#include "AdlConstraint4.h"
#include "Adl/Adl.h"

typedef adl::Buffer<Constraint4>* SolverData;
#else
typedef void* SolverData;
#endif

typedef void* ShapeDataType;


struct Contact4
{
	_MEM_ALIGNED_ALLOCATOR16;

	float4 m_worldPos[4];
	float4 m_worldNormal;
//	float m_restituitionCoeff;
//	float m_frictionCoeff;
	u16 m_restituitionCoeffCmp;
	u16 m_frictionCoeffCmp;
	int m_batchIdx;

	u32 m_bodyAPtr;
	u32 m_bodyBPtr;

	//	todo. make it safer
	int& getBatchIdx() { return m_batchIdx; }
	float getRestituitionCoeff() const { return ((float)m_restituitionCoeffCmp/(float)0xffff); }
	void setRestituitionCoeff( float c ) { ADLASSERT( c >= 0.f && c <= 1.f ); m_restituitionCoeffCmp = (u16)(c*0xffff); }
	float getFrictionCoeff() const { return ((float)m_frictionCoeffCmp/(float)0xffff); }
	void setFrictionCoeff( float c ) { ADLASSERT( c >= 0.f && c <= 1.f ); m_frictionCoeffCmp = (u16)(c*0xffff); }

	float& getNPoints() { return m_worldNormal.w; }
	float getNPoints() const { return m_worldNormal.w; }

	float getPenetration(int idx) const { return m_worldPos[idx].w; }

	bool isInvalid() const { return ((u32)m_bodyAPtr+(u32)m_bodyBPtr) == 0; }
};

struct ContactPoint4
		{
			float4 m_worldPos[4];
			union
			{
				float4 m_worldNormal;

				struct Data
				{
					int m_padding[3];
					float m_nPoints;	//	for cl
				}m_data;

			};
			float m_restituitionCoeff;
			float m_frictionCoeff;
//			int m_nPoints;
//			int m_padding0;

			void* m_bodyAPtr;
			void* m_bodyBPtr;
//			int m_padding1;
//			int m_padding2;

			float& getNPoints() { return m_data.m_nPoints; }
			float getNPoints() const { return m_data.m_nPoints; }

			float getPenetration(int idx) const { return m_worldPos[idx].w; }

//			__inline
//			void load(int idx, const ContactPoint& src);
//			__inline
//			void store(int idx, ContactPoint& dst) const;

			bool isInvalid() const { return ((u32)m_bodyAPtr+(u32)m_bodyBPtr) == 0; }

		};


#endif //ADL_CONTACT4_H


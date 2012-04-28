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
//Originally written by Takahiro Harada

#ifndef CH_NARROW_PHASE_H
#define CH_NARROW_PHASE_H



#include "AdlMath.h"
#include "AdlContact4.h"
#include "AdlRigidBody.h"

#include "../ConvexHeightFieldShape.h"


#include "../../../opencl/broadphase_benchmark/btOpenCLArray.h"

class ShapeBase;

class ChNarrowphaseBase
{
	public:
		struct Config
		{
			float m_collisionMargin;
		};

};

class ChNarrowphase : public ChNarrowphaseBase
{
	public:

		cl_kernel	m_supportCullingKernel;
		cl_kernel	m_narrowphaseKernel;
		cl_kernel	m_narrowphaseWithPlaneKernel;

		btOpenCLArray<unsigned int>* m_counterBuffer;


		cl_context m_context;
		cl_device_id m_device;
		cl_command_queue m_queue;


		enum
		{
			N_TASKS = 4,
			HEIGHT_RES = ConvexHeightField::HEIGHT_RES,
		};

		struct ShapeData
		{
			float4 m_normal[HEIGHT_RES*HEIGHT_RES*6];
			u32 m_height4[HEIGHT_RES*HEIGHT_RES*6];
			u32 m_supportHeight4[HEIGHT_RES*HEIGHT_RES*6];

			float m_scale;
			float m_padding0;
			float m_padding1;
			float m_padding2;
		};

		struct ConstData
		{
			int m_nPairs;
			float m_collisionMargin;
			int m_capacity;
			int m_paddings[1];
		};
		
		ChNarrowphase(cl_context ctx, cl_device_id device, cl_command_queue queue);
		virtual ~ChNarrowphase();


		void setShape( btOpenCLArray<ShapeData>* shapeBuf, ShapeBase* shape, int idx, float collisionMargin = 0.f );
		
		void setShape( btOpenCLArray<ShapeData>* shapeBuf, ConvexHeightField* cvxShape, int idx, float collisionMargin = 0.f );

		// Run NarrowphaseKernel
		//template<bool USE_OMP>

		void execute( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ShapeData>* shapeBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const Config& cfg );

		// Run NarrowphaseWithPlaneKernel
		//template<bool USE_OMP>

		void execute( const btOpenCLArray<int2>* pairs, int nPairs, 
			const btOpenCLArray<RigidBodyBase::Body>* bodyBuf, const btOpenCLArray<ShapeData>* shapeBuf,
			const btOpenCLArray<float4>* vtxBuf, const btOpenCLArray<int4>* idxBuf,
			btOpenCLArray<Contact4>* contactOut, int& nContacts, const Config& cfg );

		// Run SupportCullingKernel
		//template<bool USE_OMP>

		int culling( const btOpenCLArray<int2>* pairs, int nPairs, const btOpenCLArray<RigidBodyBase::Body>* bodyBuf,
			const btOpenCLArray<ShapeData>* shapeBuf, const btOpenCLArray<int2>* pairsOut, const Config& cfg );
};



#endif //CH_NARROW_PHASE_H

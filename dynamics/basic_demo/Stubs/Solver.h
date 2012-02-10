#pragma once
#ifndef __ADL_SOLVER_H
#define __ADL_SOLVER_H


#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>
#include <AdlPrimitives/Search/BoundSearch.h>
#include <AdlPrimitives/Sort/RadixSort.h>
#include <AdlPrimitives/Scan/PrefixScan.h>
#include <AdlPrimitives/Sort/RadixSort32.h>

//#include <AdlPhysics/TypeDefinition.h>
#include "AdlRigidBody.h"
#include "AdlContact4.h"

//#include "AdlPhysics/Batching/Batching.h>


#define MYF4 float4
#define MAKE_MYF4 make_float4

//#define MYF4 float4sse
//#define MAKE_MYF4 make_float4sse

#include "AdlConstraint4.h"

namespace adl
{
class SolverBase
{
	public:
		

		struct ConstraintData
		{
			ConstraintData(): m_b(0.f), m_appliedRambdaDt(0.f) {}

			float4 m_linear; // have to be normalized
			float4 m_angular0;
			float4 m_angular1;
			float m_jacCoeffInv;
			float m_b;
			float m_appliedRambdaDt;

			u32 m_bodyAPtr;
			u32 m_bodyBPtr;

			bool isInvalid() const { return ((u32)m_bodyAPtr+(u32)m_bodyBPtr) == 0; }
			float getFrictionCoeff() const { return m_linear.w; }
			void setFrictionCoeff(float coeff) { m_linear.w = coeff; }
		};

		struct ConstraintCfg
		{
			ConstraintCfg( float dt = 0.f ): m_positionDrift( 0.005f ), m_positionConstraintCoeff( 0.2f ), m_dt(dt), m_staticIdx(-1) {}

			float m_positionDrift;
			float m_positionConstraintCoeff;
			float m_dt;
			bool m_enableParallelSolve;
			float m_averageExtent;
			int m_staticIdx;
		};

		static
		__inline
		Buffer<Contact4>* allocateContact4( const Device* device, int capacity )
		{
			return new Buffer<Contact4>( device, capacity );	
		}

		static
		__inline
		void deallocateContact4( Buffer<Contact4>* data ) { delete data; }

		static
		__inline
		SolverData allocateConstraint4( const Device* device, int capacity )
		{
			return new Buffer<Constraint4>( device, capacity );
		}

		static
		__inline
		void deallocateConstraint4( SolverData data ) { delete (Buffer<Constraint4>*)data; }

		static
		__inline
		void* allocateFrictionConstraint( const Device* device, int capacity, u32 type = 0 )
		{
			return 0;
		}

		static
		__inline
		void deallocateFrictionConstraint( void* data ) 
		{
		}

		enum
		{
			N_SPLIT = 16,
			N_BATCHES = 4,
			N_OBJ_PER_SPLIT = 10,
			N_TASKS_PER_BATCH = N_SPLIT*N_SPLIT,
		};
};

template<DeviceType TYPE>
class Solver : public SolverBase
{
	public:
		typedef Launcher::BufferInfo BufferInfo;

		struct Data
		{
			Data() : m_nIterations(4){}

			const Device* m_device;
			void* m_parallelSolveData;
			int m_nIterations;
			Kernel* m_batchingKernel;
			Kernel* m_batchSolveKernel;
			Kernel* m_contactToConstraintKernel;
			Kernel* m_setSortDataKernel;
			Kernel* m_reorderContactKernel;
			Kernel* m_copyConstraintKernel;
			//typename RadixSort<TYPE>::Data* m_sort;
			typename RadixSort32<TYPE>::Data* m_sort32;
			typename BoundSearch<TYPE>::Data* m_search;
			typename PrefixScan<TYPE>::Data* m_scan;
			Buffer<SortData>* m_sortDataBuffer;
			Buffer<Contact4>* m_contactBuffer;
		};

		enum
		{
			DYNAMIC_CONTACT_ALLOCATION_THRESHOLD = 2000000,
		};

		static
		Data* allocate( const Device* device, int pairCapacity );

		static
		void deallocate( Data* data );

		static
		void reorderConvertToConstraints( Data* data, const Buffer<RigidBodyBase::Body>* bodyBuf, 
		const Buffer<RigidBodyBase::Inertia>* shapeBuf, 
			Buffer<Contact4>* contactsIn, SolverData contactCOut, void* additionalData, 
			int nContacts, const ConstraintCfg& cfg );

		static
		void solveContactConstraint( Data* data, const Buffer<RigidBodyBase::Body>* bodyBuf, const Buffer<RigidBodyBase::Inertia>* inertiaBuf, 
			SolverData constraint, void* additionalData, int n );

//		static
//		int createSolveTasks( int batchIdx, Data* data, const Buffer<RigidBodyBase::Body>* bodyBuf, const Buffer<RigidBodyBase::Inertia>* shapeBuf, 
//			SolverData constraint, int n, ThreadPool::Task* tasksOut[], int taskCapacity );


		//private:
		static
		void convertToConstraints( Data* data, const Buffer<RigidBodyBase::Body>* bodyBuf, 
			const Buffer<RigidBodyBase::Inertia>* shapeBuf, 
			Buffer<Contact4>* contactsIn, SolverData contactCOut, void* additionalData, 
			int nContacts, const ConstraintCfg& cfg );

		static
		void sortContacts( Data* data, const Buffer<RigidBodyBase::Body>* bodyBuf, 
			Buffer<Contact4>* contactsIn, void* additionalData, 
			int nContacts, const ConstraintCfg& cfg );

		static
		void batchContacts( Data* data, Buffer<Contact4>* contacts, int nContacts, Buffer<u32>* n, Buffer<u32>* offsets, int staticIdx );

};

#include "Solver.inl"
#include "SolverHost.inl"
};

#undef MYF4
#undef MAKE_MYF4

#endif //__ADL_SOLVER_H

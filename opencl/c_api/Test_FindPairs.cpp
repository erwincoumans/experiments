#include "btcFindPairs.h"

#include "Test_FindPairs.h"
#include "stdio.h"

#define MAX_NUM_AABBS 10

struct MyInfo
{
	union
	{
		void* m_ptr;
		int m_value;
	};
};

int testFindPairs()
{
	btcAabbSpace aabbSpace;
	int maxNumAabbs = MAX_NUM_AABBS;
	int maxNumPairs = 100;
	int numPairs;
	unsigned char* proxyAbase;
	unsigned char* proxyBbase;
	int  proxyType;
	int pairStrideInBytes;
	btcAabbProxy proxies[MAX_NUM_AABBS];

	aabbSpace = plCreateBruteforceSpace(maxNumAabbs, maxNumPairs);

	for (int i=0;i<maxNumAabbs;i++)
	{
		float minX = 0.f;float minY = 0.f;float minZ = 0.f;
		float maxX = 0.f;float maxY = 0.f;float maxZ = 0.f;
		
		MyInfo info;
		info.m_value = i;

		proxies[i] = btcCreateAabbProxy(aabbSpace, info.m_ptr, minX,minY,minZ, maxX,maxY,maxZ);
	}
	
	numPairs = btcFindPairs(aabbSpace);

	btbBuffer pairs = btcGetPairBuffer(aabbSpace);
	//int numPairs = btbGetSize(pairs);

	btcMapPairBuffer(aabbSpace,&numPairs, &proxyAbase, &proxyBbase,&proxyType, &pairStrideInBytes);
	for (int i=0;i<numPairs;i++)
	{
		void** infoA = (void**)(proxyAbase+pairStrideInBytes*i);
		void** infoB = (void**)(proxyBbase+pairStrideInBytes*i);
		
		printf("pair[%d] = (%d,%d)\n",i,*infoA,*infoB);
	}
	btcUnmapBuffer(aabbSpace);



	plDestroySpace(aabbSpace);
	return 0;
}
#include "btcFindPairs.h"

#include "Test_FindPairs.h"

int testFindPairs()
{
	btcAabbSpace aabbSpace;
	int maxNumAabbs = 10;
	int maxNumPairs = 100;
	int numPairs;
	unsigned char* proxyAbase;
	unsigned char* proxyBbase;
	int  proxyType;
	int pairStrideInBytes;
	
	aabbSpace = plCreateBruteforceSpace(maxNumAabbs, maxNumPairs);

	btcFindPairs(aabbSpace);

	btbBuffer pairs = btcGetPairBuffer(aabbSpace);
	//int numPairs = btbGetSize(pairs);

	

	btcMapPairBuffer(aabbSpace,&numPairs, &proxyAbase, &proxyBbase,&proxyType, &pairStrideInBytes);
	btcUnmapBuffer(aabbSpace);

	plDestroySpace(aabbSpace);
	return 0;
}
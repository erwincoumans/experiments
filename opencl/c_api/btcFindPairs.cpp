
#include "btcFindPairs.h"


class btbBruteForceSpace : public btbAabbSpaceInterface
{
public:
	btbBruteForceSpace()
	{
	}

	virtual ~btbBruteForceSpace(){}
	virtual btcAabbSpace plCreateBruteforceSpace(int maxNumAabbs, int maxNumPairs)
	{
		return 0;
	}
	virtual void	plDestroySpace(btcAabbSpace bp)
	{
	}
	virtual btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
	{
		return 0;
	}
	virtual void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle)
	{
	}
	virtual void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
	{
	}
	virtual int	btcFindPairs(btcAabbSpace bp)
	{
		return 0;
	}
	virtual btbBuffer btcGetPairBuffer(btcAabbSpace bp)
	{
		return 0;
	}
	virtual void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes)
	{
	}
	virtual void btcUnmapBuffer(btcAabbSpace aabbSpace)
	{
	}
};

btcAabbSpace plCreateBruteforceSpace(int maxNumAabbs, int maxNumPairs)
{
	btbBruteForceSpace* space = new btbBruteForceSpace();
	return (btcAabbSpace) space;
}

void	plDestroySpace(btcAabbSpace bp)
{
	btbAabbSpaceInterface* space = (btbAabbSpaceInterface*)bp;
	delete space;
}

btcAabbProxy btcCreateAabbProxy(btcAabbSpace bp, void* clientData, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
{
	btcAabbProxy proxy = 0;
	return proxy;
}

void btcDestroyAabbProxy(btcAabbSpace bp, btcAabbProxy proxyHandle)
{

}

void btcSetAabb(btcAabbSpace bp, btcAabbProxy aabbHandle, float minX,float minY,float minZ, float maxX,float maxY, float maxZ)
{

}

int btcFindPairs(btcAabbSpace bp)
{
	return 0;
}

btbBuffer btcGetPairBuffer(btcAabbSpace bp)
{
	//1) we don't know if the buffer is on the host or device
	//2) we don't know the layout/size of each element in the buffer
	//3) how many total elements are in the buffer?
	btbBuffer buf = 0;
	return buf;
}


void btcMapPairBuffer(btcAabbSpace aabbSpace, int* numPairs, unsigned char** proxyAbase, unsigned char** proxyBbase,int* proxyType, int* pairStrideInBytes)
{
	*numPairs = 0;
}
void btcUnmapBuffer(btcAabbSpace aabbSpace)
{
}

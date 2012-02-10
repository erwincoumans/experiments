#include "LinearMath/btQuickprof.h"

int result=0;
void innerloop()
{
    BT_PROFILE("innerloop");
    for (int i=0;i<1000000;i++)
    {
        result += i;
    }
}

void render()
{
    BT_PROFILE("render");
    
    innerloop();

}

int main(int argc, char* argv[])
{

    while(1)
    {

        CProfileManager::Reset();
        render();
        CProfileManager::Increment_Frame_Counter();
    
        bool showStats = true;
        if (showStats)
            CProfileManager::dumpAll();    
    }    

    printf("result=%d",result);
}
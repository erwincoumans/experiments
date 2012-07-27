#ifndef _BARRIER_SUPPORT_H
#define _BARRIER_SUPPORT_H

#define NUM_THREADS 5
#define MAX_NESTED_BARRIER_LEVELS 4

struct Args
{
    int m_i[NUM_THREADS];
    int m_barrierNestingLevel[NUM_THREADS];
    int m_barrier[MAX_NESTED_BARRIER_LEVELS][NUM_THREADS];
    int m_savedBarrier[MAX_NESTED_BARRIER_LEVELS][NUM_THREADS];
    
    Args()
    {
        for (int i=0;i<NUM_THREADS;i++)
        {
            m_i[i]=0;
            m_barrierNestingLevel[i] = 0;
            for (int n=0;n<MAX_NESTED_BARRIER_LEVELS;n++)
            {
                m_barrier[n][i]=0;
                m_savedBarrier[n][i]=0;
                
            }
        }
    }
};

#define BARRIER(label)     args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId]++; \
return 1; \
label:\

#define SKIP_TO_BARRIER \
switch (args->m_barrierNestingLevel[threadId])                                      \
{                                                                                   \
case 0:                                                                             \
{                                                                               \
switch (args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId])   \
{                                                                           \
case 0:                                                                 \
break;                                                              \
case 1:                                                                 \
goto label01;                                                       \
case 2:                                                                 \
goto label02;                                                       \
case 3:                                                                 \
goto label03;                                                       \
default:                                                                \
printf("error, unknown barrier\n");                                 \
};                                                                          \
break;                                                                      \
}                                                                               \
case 1:                                                                             \
{                                                                               \
switch (args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId])   \
{                                                                           \
case 0:                                                                 \
assert(0);                                                          \
break;                                                              \
case 1:                                                                 \
goto label11;                                                       \
case 2:                                                                 \
goto label12;                                                       \
default:                                                                \
printf("error, unknown barrier\n");                                 \
};                                                                          \
break;                                                                      \
}                                                                               \
default:                                                                            \
{                                                                               \
}                                                                               \
};


#define INC_LEVEL \
args->m_barrierNestingLevel[threadId]++;\
args->m_savedBarrier[args->m_barrierNestingLevel[threadId]][threadId] = args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId];

#define DEC_LEVEL \
args->m_savedBarrier[args->m_barrierNestingLevel[threadId]][threadId] = args->m_barrier[args->m_barrierNestingLevel[threadId]][threadId]; \
args->m_barrierNestingLevel[threadId]--;


#define I (args->m_i[threadId])


#endif //_BARRIER_SUPPORT_H



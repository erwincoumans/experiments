#ifndef __SBM_H__
#define __SBM_H__

#include "GLES2/gl2.h"

typedef struct SBM_HEADER_t
{
    unsigned int magic;
    unsigned int size;
    char name[64];
    unsigned int num_attribs;
    unsigned int num_frames;
    unsigned int num_vertices;
    unsigned int num_indices;
    unsigned int index_type;
} SBM_HEADER;

typedef struct SBM_ATTRIB_HEADER_t
{
    char name[64];
    unsigned int type;
    unsigned int components;
    unsigned int flags;
} SBM_ATTRIB_HEADER;

typedef struct SBM_FRAME_HEADER_t
{
    unsigned int first;
    unsigned int count;
    unsigned int flags;
} SBM_FRAME_HEADER;

typedef struct SBM_VEC4F_t
{
    float x;
    float y;
    float z;
    float w;
} SBM_VEC4F;

class SBObject
{
public:
    SBObject(void);
    virtual ~SBObject(void);

    bool LoadFromSBM(const char * filename);
    void Render();
    bool Free(void);

    unsigned int GetAttributeCount(void) const
    {
        return m_header.num_attribs;
    }

    const char * GetAttributeName(unsigned int index) const
    {
        return index < m_header.num_attribs ? m_attrib[index].name : 0;
    }

protected:
    GLuint m_vao;
    GLuint m_attribute_buffer;
    GLuint m_index_buffer;

    GLuint m_num_attribs;
    GLuint m_num_verticies;
    
    GLuint m_vertexIndex;
    GLuint m_normalIndex;
    GLuint m_texCoord0Index;

    SBM_HEADER m_header;
    SBM_ATTRIB_HEADER * m_attrib;
    SBM_FRAME_HEADER * m_frame;

    unsigned char * m_data;
    unsigned char * m_raw_data;
};

#endif /* __SBM_H__ */

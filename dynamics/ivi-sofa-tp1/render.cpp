#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>

#include <png.h>
#ifdef _MSC_VER
#pragma comment(lib,"libpng.lib")
#pragma comment(lib,"zlib.lib")
#endif

#include <iostream>
#include <fstream>

#include "simulation.h"
#include "render.h"

//// DATA ////

RenderFlag render_flag[RENDER_NFLAGS] = {
    RenderFlag(true, "display help messages"),
    RenderFlag(true, "display statistics"),
    RenderFlag(0, 3, "display surface mesh, FEM mesh, or none"),
    RenderFlag(0, 8, "display FEM particles"),
    RenderFlag(0, 4, "display debug info"),
    RenderFlag(2, 3, "enable shaders"),
};

TCoord camera_position;
TCoord4 light0_position;
TCoord4 light1_position;
TColor light_ambient(0.2f, 0.2f, 0.2f, 1.0f);
TColor light0_color (0.75f, 0.75f, 0.9f, 1.0f);
TColor light1_color (0.4f, 0.4f ,0.4f ,1.0f);
TColor background_color (1.0f, 1.0f, 1.0f, 0.0f);

bool render_surface = true;
#ifdef __APPLE__
bool use_vbo = false;
#else
bool use_vbo = true;
#endif

bool use_shaders = true;

//// METHODS ////

GLint loadTexture(const std::string& filename)
{
    FILE *file;

	if ((file = fopen(filename.c_str(), "rb")) == NULL)
	{
		std::cerr << "File not found : " << filename << std::endl;
		return 0;
	}
	
	png_structp PNG_reader = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (PNG_reader == NULL)
	{
		std::cerr << "png_create_read_struct failed for file "<< filename << std::endl;
		fclose(file);
		return 0;
	}
	
	png_infop PNG_info = png_create_info_struct(PNG_reader);
	png_infop PNG_end_info = png_create_info_struct(PNG_reader);
	if (PNG_info == NULL || PNG_end_info == NULL)
	{
		std::cerr << "png_create_info_struct failed for file " << filename << std::endl;
		png_destroy_read_struct(&PNG_reader, NULL, NULL);
		fclose(file);
		return 0;
	}
	
	if (setjmp(png_jmpbuf(PNG_reader)))
	{
		std::cerr << "Loading failed for PNG file " << filename << std::endl;
		png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
		fclose(file);
		return 0;
	}
	
	png_init_io(PNG_reader, file);
	
	png_read_info(PNG_reader, PNG_info);
	
	png_uint_32 width, height;
	width = png_get_image_width(PNG_reader, PNG_info);
	height = png_get_image_height(PNG_reader, PNG_info);
	
	png_uint_32 bit_depth, channels, color_type;
	bit_depth = png_get_bit_depth(PNG_reader, PNG_info);
	channels = png_get_channels(PNG_reader, PNG_info);
	color_type = png_get_color_type(PNG_reader, PNG_info);
	
	std::cout << "PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
	bool changed = false;
	if (color_type == PNG_COLOR_TYPE_PALETTE)
	{
		png_set_palette_to_rgb(PNG_reader);
		changed = true;
	}
	
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
	{
#if PNG_LIBPNG_VER >= 10209
		png_set_expand_gray_1_2_4_to_8(PNG_reader);
#else
        png_set_gray_1_2_4_to_8(PNG_reader);    // deprecated from libpng 1.2.9
#endif
		changed = true;
	}
	/*
	if (bit_depth == 16)
	{
		png_set_strip_16(PNG_reader);
		changed = true;
	}
    */
	if (changed)
	{
		png_read_update_info(PNG_reader, PNG_info);
		bit_depth = png_get_bit_depth(PNG_reader, PNG_info);
		channels = png_get_channels(PNG_reader, PNG_info);
		color_type = png_get_color_type(PNG_reader, PNG_info);
		std::cout << "Converted PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
	}

    GLint internalFormat; // 1, 2, 3, or 4, or one of the following symbolic constants: GL_ALPHA, GL_ALPHA4, GL_ALPHA8, GL_ALPHA12, GL_ALPHA16, GL_COMPRESSED_ALPHA, GL_COMPRESSED_LUMINANCE, GL_COMPRESSED_LUMINANCE_ALPHA, GL_COMPRESSED_INTENSITY, GL_COMPRESSED_RGB, GL_COMPRESSED_RGBA, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT32, GL_LUMINANCE, GL_LUMINANCE4, GL_LUMINANCE8, GL_LUMINANCE12, GL_LUMINANCE16, GL_LUMINANCE_ALPHA, GL_LUMINANCE4_ALPHA4, GL_LUMINANCE6_ALPHA2, GL_LUMINANCE8_ALPHA8, GL_LUMINANCE12_ALPHA4, GL_LUMINANCE12_ALPHA12, GL_LUMINANCE16_ALPHA16, GL_INTENSITY, GL_INTENSITY4, GL_INTENSITY8, GL_INTENSITY12, GL_INTENSITY16, GL_R3_G3_B2, GL_RGB, GL_RGB4, GL_RGB5, GL_RGB8, GL_RGB10, GL_RGB12, GL_RGB16, GL_RGBA, GL_RGBA2, GL_RGBA4, GL_RGB5_A1, GL_RGBA8, GL_RGB10_A2, GL_RGBA12, GL_RGBA16, GL_SLUMINANCE, GL_SLUMINANCE8, GL_SLUMINANCE_ALPHA, GL_SLUMINANCE8_ALPHA8, GL_SRGB, GL_SRGB8, GL_SRGB_ALPHA, or GL_SRGB8_ALPHA8.
    GLenum format; // GL_COLOR_INDEX, GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA, GL_RGB, GL_BGR, GL_RGBA, GL_BGRA, GL_LUMINANCE, and GL_LUMINANCE_ALPHA.
    GLenum type; // GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT, GL_INT, GL_FLOAT, GL_UNSIGNED_BYTE_3_3_2, GL_UNSIGNED_BYTE_2_3_3_REV, GL_UNSIGNED_SHORT_5_6_5, GL_UNSIGNED_SHORT_5_6_5_REV, GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_SHORT_4_4_4_4_REV, GL_UNSIGNED_SHORT_5_5_5_1, GL_UNSIGNED_SHORT_1_5_5_5_REV, GL_UNSIGNED_INT_8_8_8_8, GL_UNSIGNED_INT_8_8_8_8_REV, GL_UNSIGNED_INT_10_10_10_2, and GL_UNSIGNED_INT_2_10_10_10_REV.
    switch (bit_depth)
    {
    case 8:
        type = GL_UNSIGNED_BYTE;
        break;
    case 16:
        type = GL_UNSIGNED_SHORT;
        break;
    default:
        std::cerr << "PNG: in " << filename << ", unsupported bit depth: " << bit_depth << std::endl;
        return 0;
    }
    switch (channels)
    {
    case 1:
        format = GL_LUMINANCE;
        internalFormat = (bit_depth <= 8) ? GL_INTENSITY8 : GL_INTENSITY16;
        break;
    case 2:
        format = GL_LUMINANCE_ALPHA;
        internalFormat = (bit_depth <= 8) ? GL_LUMINANCE8_ALPHA8 : GL_LUMINANCE16_ALPHA16;
        break;
    case 3:
        format = GL_RGB;
        internalFormat = (bit_depth <= 8) ? GL_RGB8 : GL_RGB16;
        break;
    case 4:
        format = GL_RGBA;
        internalFormat = (bit_depth <= 8) ? GL_RGBA8 : GL_RGBA16;
        break;
    default:
        std::cerr << "PNG: in " << filename << ", unsupported number of channels: " << channels << std::endl;
        return 0;
    }

    unsigned int lineSize = width * ((channels*bit_depth+7)/8);
    // align to 4 bytes
    lineSize = (lineSize + 3) & -4;
    png_byte* data = (png_byte*)malloc(height*lineSize);
	png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));
	for (png_uint_32 row = 0; row < height; ++row)
        PNG_rows[height - 1 - row] = data+row*lineSize;
	
	png_read_image(PNG_reader, PNG_rows);
	
	free(PNG_rows);
	
	png_read_end(PNG_reader, PNG_end_info);
	
	png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
	fclose(file);
	
    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#ifdef SOFA_HAVE_GLEW
    if (GLEW_ARB_framebuffer_object || GLEW_EXT_framebuffer_object)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    else
#endif
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
    free(data);
#ifdef SOFA_HAVE_GLEW
    if (GLEW_ARB_framebuffer_object)
        glGenerateMipmap(GL_TEXTURE_2D);
    else if (GLEW_EXT_framebuffer_object)
        glGenerateMipmapEXT(GL_TEXTURE_2D);
#endif
	return textureID;
}

extern std::string parentDir;

GLint textureColor = 0;
GLint textureNormal = 0;

GLuint shaderVertex = 0;
GLuint shaderFragment = 0;
GLuint shaderProgram = 0;
GLuint shaderTangentAttrib = 0;
GLuint shaderColorMap = 0;
GLuint shaderNormalMap = 0;



std::string loadTextFile(const std::string& filename)
{
	// Open the file passed in
	std::ifstream fin(filename.c_str());

	// Make sure we opened the file correctly
	if(!fin)
		return "";

	std::string strLine = "";
	std::string strText = "";

	// Go through and store each line in the text file within a "string" object
	while(std::getline(fin, strLine))
	{
		strText += "\n" + strLine;
	}

	// Close our file
	fin.close();

	// Return the text file's data
	return strText;
}

GLint loadShader(GLint target, const std::string& filename)
{
    std::string source = loadTextFile(filename);
    if (source.empty()) return 0;
    GLint shader = glCreateShaderObjectARB(target);
    const char* src = source.c_str();
    glShaderSourceARB(shader, 1, &src, NULL);
    glCompileShaderARB(shader);

    GLint compiled = 0, length = 0, laux = 0;
    glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compiled);
    if (!compiled) std::cerr << "ERROR: Compilation of "<<filename<<" shader failed:"<<std::endl;
    else std::cout << "Compilation of "<<filename<<" shader OK" << std::endl;

    glGetObjectParameterivARB(shader, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length > 1)
    {
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(shader, length, &laux, logString);
        std::cerr << logString << std::endl;
        free(logString);
    }
    return shader;
}



GLUquadric* pGLUquadric = NULL;

void initgl()
{
    { // no surface -> display points
        render_flag[RENDER_FLAG_MESH].nvalues = (render_meshes.empty()) ? 2 : 3;
        render_flag[RENDER_FLAG_POINT]        = (render_meshes.empty()) ? 1 : 0;
    }

    light0_position = simulation_center + TCoord(0.0f,2.0f,0.0f)*simulation_size; light0_position[3] = 1;
    light1_position = simulation_center + TCoord(3.0f,-0.5f,-1.0f)*simulation_size; light1_position[3] = 1;
    reset_camera();

#ifdef SOFA_HAVE_GLEW
    use_vbo = use_vbo && GLEW_VERSION_1_5; // || GLEW_ARB_vertex_buffer_object;
    //use_shaders = use_shaders && GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader && GLEW_ARB_shader_objects && GLEW_ARB_shading_language_100;
#else
    // we assume OpenGL version is at least 1.5, since CUDA is supported...
#endif

    pGLUquadric = gluNewQuadric();

    glDepthFunc(GL_LEQUAL);
    glClearDepth(1);
    glEnable(GL_DEPTH_TEST);
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glClearColor ( background_color[0], background_color[1], background_color[2], background_color[3] );

    GLfloat    vzero[4] = {0, 0, 0, 0};
    GLfloat    vone[4] = {1, 1, 1, 1};
    
    // Set light model
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient.ptr());
    
    // Setup 'light 0'
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, vzero);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_color.ptr());
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0_color.ptr());
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position.ptr());
    
    // Setup 'light 1'
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, vzero);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_color.ptr());
    glLightfv(GL_LIGHT1, GL_SPECULAR, light1_color.ptr());
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position.ptr());
    
    // Enable color tracking
    glMaterialfv(GL_FRONT, GL_AMBIENT, vone);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, vone);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glShadeModel(GL_SMOOTH);

    // disable specular
    //glMaterialfv(GL_FRONT, GL_SPECULAR, vzero);
    glMaterialfv(GL_FRONT, GL_SPECULAR, vone);
    glMateriali(GL_FRONT, GL_SHININESS, 100);

    glEnableClientState(GL_VERTEX_ARRAY);
    //glDisableClientState(GL_NORMAL_ARRAY);

    if (render_meshes.size() >= 1 && !render_meshes[0]->texcoords.empty())
    {
        textureColor = loadTexture(parentDir + "data/colorMap.png");
        if (textureColor != 0)
        {
            render_meshes[0]->textureFilename = parentDir + "data/colorMap.png";
            render_meshes[0]->color = TColor(1,1,1,1);
        }
        if (use_shaders && textureColor != 0)
        {
            textureNormal = loadTexture(parentDir + "data/normalMap.png");
            if (textureNormal != 0)
            {
                render_meshes[0]->computeTangents = true;
                render_meshes[0]->updateNormals();
            }
        }
    }
    shaders_reload();
}

void shaders_reload()
{
    if (!use_shaders)
    {
        std::cout << "Shaders disabled" << std::endl;
    }
    else if (textureNormal == 0)
    {
        std::cout << "No normalmap texture" << std::endl;
    }
    else
    {
        GLint oldShaderVertex = shaderVertex;
        GLint oldShaderFragment = shaderFragment;
        GLint oldShaderProgram = shaderProgram;
        shaderVertex = loadShader(GL_VERTEX_SHADER_ARB, parentDir + "data/bumpVertexShader.glsl");
        shaderFragment = loadShader(GL_FRAGMENT_SHADER_ARB, parentDir + "data/bumpFragmentShader.glsl");
        if (shaderVertex && shaderFragment)
        {
            shaderProgram = glCreateProgramObjectARB();	
            /* use program object */
            glAttachObjectARB(shaderProgram, shaderVertex);
            glAttachObjectARB(shaderProgram, shaderFragment);

            /* link */
            glLinkProgramARB(shaderProgram);
            GLint status = 0, length = 0, laux = 0;
            glGetObjectParameterivARB(shaderProgram, GL_OBJECT_LINK_STATUS_ARB, &status);
            if (!status) std::cerr << "ERROR: Link of shaders failed:"<<std::endl;
            else std::cout << "Link of shaders OK" << std::endl;
            glGetObjectParameterivARB(shaderProgram, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
            if (length > 1)
            {
                GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
                glGetInfoLogARB(shaderProgram, length, &laux, logString);
                std::cerr << logString << std::endl;
                free(logString);
            }
            if (!status) shaderProgram = 0;
        }
        if (shaderProgram) // load successfull
        {
            if (oldShaderVertex) glDeleteObjectARB(oldShaderVertex);
            if (oldShaderFragment) glDeleteObjectARB(oldShaderFragment);
            if (oldShaderProgram) glDeleteObjectARB(oldShaderProgram);
            glUseProgramObjectARB(shaderProgram);
            shaderTangentAttrib = glGetAttribLocationARB(shaderProgram,"tangent");
            shaderColorMap = glGetUniformLocationARB(shaderProgram,"colorMap");
            shaderNormalMap = glGetUniformLocationARB(shaderProgram,"normalMap");
            glUniform1iARB(shaderColorMap,0); 
            glUniform1iARB(shaderNormalMap,1); 
            glUseProgramObjectARB(0);
        }
        else // keep previous
        {
            if (shaderVertex) glDeleteObjectARB(shaderVertex);
            shaderVertex = oldShaderVertex;
            if (shaderFragment) glDeleteObjectARB(shaderFragment);
            shaderFragment = oldShaderFragment;
            if (shaderProgram) glDeleteObjectARB(shaderProgram);
            shaderProgram = oldShaderProgram;
        }
    }
}

//// CUDA / OPENGL INTEROP

template<class T>
GLenum myglType()
{
    GLenum vtype = GL_FLOAT; // default type
    if (sofa::defaulttype::DataTypeInfo<T>::Integer)
    {
        switch(sizeof(typename sofa::defaulttype::DataTypeInfo<T>::ValueType))
        {
        case sizeof(char): vtype = GL_UNSIGNED_BYTE; break;
        case sizeof(short): vtype = GL_UNSIGNED_SHORT; break;
        case sizeof(int): vtype = GL_UNSIGNED_INT; break;
        }
    }
    else if (sofa::defaulttype::DataTypeInfo<T>::Scalar)
    {
        switch(sizeof(typename sofa::defaulttype::DataTypeInfo<T>::ValueType))
        {
        case sizeof(float): vtype = GL_FLOAT; break;
        case sizeof(double): vtype = GL_DOUBLE; break;
        }
    }
    return vtype;
}

template<class T>
void myglVertexPointer(sofa::helper::vector<T,MyMemoryManager<T> >& x)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glVertexPointer(sofa::defaulttype::DataTypeInfo<T>::size(), myglType<T>(), sizeof(T), pointer);
}

template<class T>
void myglNormalPointer(sofa::helper::vector<T,MyMemoryManager<T> >& x)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glNormalPointer(myglType<T>(), sizeof(T), pointer);
}

template<class T>
void myglColorPointer(sofa::helper::vector<T,MyMemoryManager<T> >& x)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glColorPointer(sofa::defaulttype::DataTypeInfo<T>::size(), myglType<T>(), sizeof(T), pointer);
}

template<class T>
void myglTexCoordPointer(sofa::helper::vector<T,MyMemoryManager<T> >& x)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glTexCoordPointer(sofa::defaulttype::DataTypeInfo<T>::size(), myglType<T>(), sizeof(T), pointer);
}

template<class T>
void myglVertexAttribPointerARB(GLint attrib, sofa::helper::vector<T,MyMemoryManager<T> >& x, bool normalized = false)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glVertexAttribPointerARB(attrib, sofa::defaulttype::DataTypeInfo<T>::size(), myglType<T>(), normalized ? GL_TRUE:GL_FALSE, sizeof(T), pointer);
}

template<class T>
void myglDrawElements(GLenum mode, sofa::helper::vector<T,MyMemoryManager<T> >& x)
{
    const GLvoid * pointer = NULL;
#ifdef SOFA_DEVICE_CUDA
    GLuint vbo_x = 0;
    if (use_vbo && MyMemoryManager<T>::SUPPORT_GL_BUFFER)
        vbo_x = x.bufferRead(true);
    if (vbo_x)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_x);
        pointer = NULL;
    }
    else
#endif
    {
        pointer = x.hostRead();
    }
    glDrawElements(mode, x.size() * sofa::defaulttype::DataTypeInfo<T>::size(), (sizeof(typename sofa::defaulttype::DataTypeInfo<T>::ValueType)==sizeof(char)) ? GL_UNSIGNED_BYTE : (sizeof(typename sofa::defaulttype::DataTypeInfo<T>::ValueType)==sizeof(short)) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, pointer);
}

//// MAIN METHOD ////

void render()
{
    render_surface = (render_meshes.empty())? false : render_flag[RENDER_FLAG_MESH] == 0;
    bool render_fem_tetra = (render_flag[RENDER_FLAG_MESH] == ((render_meshes.empty())? 0 : 1));
    TColor bgcolor;
    if (render_flag[RENDER_FLAG_DEBUG] == 3)
        bgcolor = TColor(0.0f,0.0f,0.0f,0.0f);
    else
        bgcolor = TColor(1.0f,1.0f,1.0f,0.0f);
    if (bgcolor != background_color)
    {
        background_color = bgcolor;
        glClearColor ( background_color[0], background_color[1], background_color[2], background_color[3] );
        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }
    if (render_surface)
        simulation_mapping(); // make sure the surfaces are up-to-date

    camera_position = camera_lookat + camera_direction * camera_distance;
    glMatrixMode   ( GL_MODELVIEW );  // Select The Model View Matrix
    glLoadIdentity ( );               // Reset The Model View Matrix
    gluLookAt(camera_position[0], camera_position[1], camera_position[2],
              camera_lookat[0], camera_lookat[1], camera_lookat[2],
              0, 1, 0);

    glLightfv(GL_LIGHT0, GL_POSITION, light0_position.ptr());
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position.ptr());


    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    if (fem_mesh)
    {
        FEMMesh* mesh = fem_mesh;
        if (render_flag[RENDER_FLAG_POINT]&3)
        {
            myglVertexPointer(mesh->positions);
            if (!render_surface)
            {
                glColor3f(0.1f,0.1f,0.1f);
                glPointSize(2*(render_flag[RENDER_FLAG_POINT]&3));
                glDrawArrays(GL_POINTS, 0, mesh->positions.size());
                glPointSize(1);
            }

            if (!fem_mesh->fixedParticles.empty())
            {
                glColor3f(1.0f,0.4f,0.4f);
                glPointSize(4*(render_flag[RENDER_FLAG_POINT]&3));
                myglDrawElements(GL_POINTS, mesh->fixedParticles);
                glPointSize(1);
            }
        }

        if (render_fem_tetra)
        {
            FEMMesh* mesh = fem_mesh;

            const TCoord* x = mesh->positions.hostRead();
            const TTetra* tetrahedra = mesh->tetrahedra.hostRead();
            glBegin(GL_TRIANGLES);
            for (unsigned int i=0;i<mesh->tetrahedra.size();++i)
            {
                TTetra t = tetrahedra[i];
                TReal youngModulus = mesh->tetraYoungModulus(i, &simulation_params);
                TReal color = (TReal)(((youngModulus / simulation_params.youngModulusTop)-1.1)*0.1f);
                if (color < 0) color = 0; else if (color > 1) color = 1;
                TCoord a = x[t[0]];
                TCoord b = x[t[1]];
                TCoord c = x[t[2]];
                TCoord d = x[t[3]];
                TCoord center = (a+b+c+d)*(TReal)0.125;
                a = (a+center)*(TReal)0.666667;
                b = (b+center)*(TReal)0.666667;
                c = (c+center)*(TReal)0.666667;
                d = (d+center)*(TReal)0.666667;
                
                glColor4f(0+color,0,1-color,1);
                glVertex3fv(a.ptr()); glVertex3fv(b.ptr()); glVertex3fv(c.ptr());
                
                glColor4f(0+color,0.5f-0.5f*color,1-color,1);
                glVertex3fv(b.ptr()); glVertex3fv(c.ptr()); glVertex3fv(d.ptr());

                glColor4f(0+color,1-color,1-color,1);
                glVertex3fv(c.ptr()); glVertex3fv(d.ptr()); glVertex3fv(a.ptr());

                glColor4f(0.5f+0.5f*color,1-color,1-0.5f*color,1);
                glVertex3fv(d.ptr()); glVertex3fv(a.ptr()); glVertex3fv(b.ptr());
            }
            glEnd(); // GL_TRIANGLES
        }
        if (render_flag[RENDER_FLAG_DEBUG]==1)
        {
            if (mesh->velocity.size() == mesh->positions.size())
            {
                glColor3f(1,1,0);
                glBegin(GL_LINES);
                for (unsigned int i=0;i<mesh->positions.size();++i)
                {
                    TCoord x = mesh->positions[i];
                    TCoord x2 = x + mesh->velocity[i]*0.1;
                    glVertex3fv(x.ptr()); glVertex3fv(x2.ptr());
                }
                glEnd(); // GL_LINES
            }
            if (mesh->a.size() == mesh->positions.size())
            {
                glColor3f(1,0,0);
                glBegin(GL_LINES);
                for (unsigned int i=0;i<mesh->positions.size();++i)
                {
                    TCoord x = mesh->positions[i] + mesh->velocity[i]*0.1;
                    TCoord x2 = x + mesh->a[i]*0.1;
                    glVertex3fv(x.ptr()); glVertex3fv(x2.ptr());
                }
                glEnd(); // GL_LINES
            }
        }

        if (mesh->externalForce.index != -1 && !(render_flag[RENDER_FLAG_POINT]&4))
        {
            TCoord p1 = mesh->positions[mesh->externalForce.index];
            TCoord p2 = p1 + mesh->externalForce.value * 0.001;
            glLineWidth(3);
            glBegin(GL_LINES);
            glColor3f(0.8f,0.2f,0.2f); glVertex3fv(p1.ptr());
            glColor3f(1.0f,0.6f,0.6f); glVertex3fv(p2.ptr());
            glEnd(); // GL_LINES
            glLineWidth(1);
        }
    }

    if (!render_meshes.empty() && render_surface)
    {
        glEnable(GL_LIGHTING);
        glEnable(GL_CULL_FACE);
        for (unsigned int mi=0;mi<render_meshes.size();++mi)
        {
            SurfaceMesh* mesh = render_meshes[mi];
            const bool tex = render_flag[RENDER_FLAG_SHADERS]>=1 && ((GLint)textureColor != 0) && !mesh->texcoords.empty();
            const bool texnormal = render_flag[RENDER_FLAG_SHADERS]>=2 && shaderProgram && shaderNormalMap && ((GLint)textureNormal != 0) && shaderTangentAttrib && !mesh->tangents.empty();
            if (tex)       glColor3f(1.0f, 1.0f, 1.0f);
            else if (mi&1) glColor3f(0.15f, 0.15f, 0.15f);
            else           glColor3f(0.8f, 0.8f, 0.8f);
            myglVertexPointer(mesh->positions);
            myglNormalPointer(mesh->normals);
            glEnableClientState(GL_NORMAL_ARRAY);
            if (tex)
            {
                myglTexCoordPointer(mesh->texcoords);
                glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                glBindTexture(GL_TEXTURE_2D, textureColor);
                glEnable(GL_TEXTURE_2D);
            }
            if (texnormal)
            {
                myglVertexAttribPointerARB(shaderTangentAttrib, mesh->tangents);
                glEnableVertexAttribArrayARB(shaderTangentAttrib);
                glActiveTextureARB(GL_TEXTURE1_ARB);
                glBindTexture(GL_TEXTURE_2D, textureNormal);
                glEnable(GL_TEXTURE_2D);
                glUseProgramObjectARB(shaderProgram);
                glUniform1iARB(shaderColorMap,0); 
                glUniform1iARB(shaderNormalMap,1); 
            }
            myglDrawElements(GL_TRIANGLES, mesh->triangles);
            if (texnormal)
            {
                glUseProgramObjectARB(0);
                glDisable(GL_TEXTURE_2D);
                glActiveTextureARB(GL_TEXTURE0_ARB);
                glDisableVertexAttribArrayARB(shaderTangentAttrib);
            }
            glDisableClientState(GL_NORMAL_ARRAY);
            if (tex)
            {
                glDisableClientState(GL_TEXTURE_COORD_ARRAY);
                glDisable(GL_TEXTURE_2D);
            }

        }
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
    }

    if (render_flag[RENDER_FLAG_DEBUG] < 2 && plane_size > 0)
    {
        glEnable(GL_LIGHTING);
        glEnable(GL_CULL_FACE);
        glColor3f(0.65f,0.5f,0.33f);
        glBegin(GL_QUADS);
        glNormal3f(0,1,0);
        for (int x=-10;x<10;++x)
            for (int z=-10; z<10;++z)
            {
                glVertex3f(plane_position[0]+(x  )*plane_size/10,plane_position[1],plane_position[2]+(z  )*plane_size/10);
                glVertex3f(plane_position[0]+(x  )*plane_size/10,plane_position[1],plane_position[2]+(z+1)*plane_size/10);
                glVertex3f(plane_position[0]+(x+1)*plane_size/10,plane_position[1],plane_position[2]+(z+1)*plane_size/10);
                glVertex3f(plane_position[0]+(x+1)*plane_size/10,plane_position[1],plane_position[2]+(z  )*plane_size/10);
            }
        glEnd(); // GL_QUADS
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
    }

    if (render_flag[RENDER_FLAG_DEBUG] < 2 && sphere_radius > 0 && simulation_params.sphereRepulsion != 0)
    {
        glEnable(GL_LIGHTING);
        glEnable(GL_CULL_FACE);
        glColor3f(1.0f,0.33f,1.0f);
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

        glPushMatrix();
        glTranslated(sphere_position[0],sphere_position[1],sphere_position[2]);
        gluSphere(pGLUquadric, sphere_radius, 16, 16);
        glPopMatrix();

        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);
    }

    if (render_flag[RENDER_FLAG_DEBUG]==1)
    {
        // axis
        glBegin(GL_LINES);
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0);
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0);
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
        glEnd();
        // lights
        glPointSize(10);
        glBegin(GL_POINTS);
        glColor3fv(light0_color.ptr()); glVertex3fv(light0_position.ptr());
        glColor3fv(light1_color.ptr()); glVertex3fv(light1_position.ptr());
        glEnd(); // GL_POINTS
        glPointSize(1);

        for (unsigned int mi=0;mi<render_meshes.size();++mi)
        {
            SurfaceMesh* mesh = render_meshes[mi];

            const TCoord* positions = mesh->positions.hostRead();
            const TCoord* normals = mesh->normals.empty() ? NULL : mesh->normals.hostRead();
            const TCoord* tangents = mesh->tangents.empty() ? NULL : mesh->tangents.hostRead();
            //const TTetra* triangles = mesh->triangles.hostRead();
            glBegin(GL_LINES);
            for (unsigned int i=0;i<mesh->positions.size();++i)
            {
                TCoord p = positions[i];
                if (normals)
                {
                    TCoord pn = p + normals[i] * (simulation_size*0.002);
                    glColor3f(0,0,1);
                    glVertex3fv(p.ptr());
                    glVertex3fv(pn.ptr());
                }
                if (tangents)
                {
                    TCoord pt = p + tangents[i] * (simulation_size*0.002);
                    glColor3f(0,1,0);
                    glVertex3fv(p.ptr());
                    glVertex3fv(pt.ptr());
                }
            }
            glEnd(); // GL_LINES
        }
    }
}

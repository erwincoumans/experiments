#include "simulation.h"
#include "render.h"
#include <sofa/helper/system/thread/CTime.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <direct.h>
#endif
#if defined (__APPLE__)
#include <sys/param.h>
#include <mach-o/dyld.h>
//#include <CoreFoundation/CoreFoundation.h>
#endif
#include <string.h>
#include <iostream>

extern void init_glut(int* argc, char** argv);
extern void setup_glut();
extern void run_glut();

extern int cuda_device;
extern bool use_vbo;


/// Get the full path of the current process. The given filename should be the value of argv[0].
std::string getProcessFullPath(const char* filename)
{
    //return "/Users/allard/work/sofa-gpu/papers/gpugems-fem/code/";
#if defined (WIN32)
    if (!filename || !filename[0])
    {
        //return __argv[0];
        int n=0;
	LPWSTR wpath = *CommandLineToArgvW(GetCommandLineW(),&n);
	if (wpath)
	{
	    char path[1024];
	    memset(path,0,sizeof(path));
	    wcstombs(path, wpath, sizeof(path)-1);
	    //std::cout << "Current process: "<<path<<std::endl;
	    if (path[0]) return path;
	}
    }
#elif defined (__linux__)
    if (!filename || filename[0]!='/')
    {
		char path[1024];
		memset(path,0,sizeof(path));
		if (readlink("/proc/self/exe",path,sizeof(path)-1) == -1)
		  std::cerr <<"Error: can't read the contents of the link." << std::endl;
		if (path[0])
			return path;
		else
			std::cout << "ERROR: can't get current process path..." << std::endl;
    }
#elif defined (__APPLE__)
    if (!filename || filename[0]!='/')
    {
        char* path = new char[4096];
		uint32_t size = 4096;
		if ( _NSGetExecutablePath( path, &size ) != 0)
        {
            delete [] path;
            path = new char[size];
             _NSGetExecutablePath( path, &size );
        }
        std::string finalPath(path);
        delete [] path;
		return finalPath;
	}
#endif

    return filename;
}

std::string getParentDir(std::string path)
{
    std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
        return ""; // no directory
    else
        return path.substr(0,pos);
}

std::string parentDir;

    int benchmark = 0;
    bool reorder = false;
    std::string fem_filename;
    std::vector<std::string> render_filenames;

int main_load();

int main(int argc, char **argv)
{
    parentDir = getProcessFullPath(argv[0]);
    parentDir = getParentDir(parentDir);
//#ifdef __APPLE__
//    if (CFBundleGetMainBundle() == ???) // we are inside a bundle -> data files are in Resources directory
//        parentDir = getParentDir(parentDir) + std::string("/Resources");
//#endif
    if (!parentDir.empty()) parentDir += '/';
    fem_filename = parentDir + "data/raptor-8418.mesh";
    render_filenames.push_back(parentDir + "data/raptor-skin.obj");
    render_filenames.push_back(parentDir + "data/raptor-misc.obj");
    bool onlyfiles = false;
    bool renderfiles = false;
    int arg = 1;
    while (argc > arg)
    {
        if (onlyfiles || argv[arg][0] != '-')
        {
            if (!renderfiles)
            {
                fem_filename = argv[arg];
                render_filenames.clear();
                renderfiles = true;
                ++arg;
            }
            else
            {
                render_filenames.push_back(argv[arg]);
                ++arg;
            }
        }
        else if (!strncmp(argv[arg],"--device=",9))
        {
            cuda_device=atoi(argv[arg]+9);
            ++arg;
        }
        else if (!strncmp(argv[arg],"--bench",7))
        {
            std::cout << "Benchmark mode." << std::endl;
            if (argv[arg][7] == '=') benchmark = atoi(argv[arg]+8);
            else benchmark = 1000;
            ++arg;
        }
        else if (!strcmp(argv[arg],"--reorder"))
        {
            reorder = true;
            ++arg;
        }
        else if (!strcmp(argv[arg],"--noreorder"))
        {
            reorder = false;
            ++arg;
        }
        else if (!strcmp(argv[arg],"--novbo"))
        {
            use_vbo = false;
            ++arg;
        }
        else if (!strcmp(argv[arg],"--vbo"))
        {
            use_vbo = true;
            ++arg;
        }
        else if (!strcmp(argv[arg],"--"))
        {
            onlyfiles = true;
            ++arg;
        }
        else
        {
            std::cerr << "Unknown option " << argv[arg] << std::endl;
            return 1;
        }
    }

    if (!simulation_preload())
        return 1;

    if (!benchmark)
    {
        std::cout << "Init GLUT" << std::endl;
        init_glut(&argc, argv);
        run_glut();
    }
    else
    {
        if (main_load()) return 1;
        sofa::helper::system::thread::ctime_t t0, t1;
        sofa::helper::system::thread::CTime::getRefTime();

        std::cout << "Running first timestep" << std::endl;
        simulation_animate();


        std::cout << "Running " << benchmark << " timesteps..." << std::endl;
        t0 = sofa::helper::system::thread::CTime::getRefTime();
        for (int it = 0; it < benchmark; ++it)
            simulation_animate();
        t1 = sofa::helper::system::thread::CTime::getRefTime();
        double dt = ((t1-t0)/(sofa::helper::system::thread::CTime::getRefTicksPerSec()*(double)benchmark));
        std::cout << "Time: " << dt*1000.0 <<" ms / timestep, " << 1.0/dt << " FPS." << std::endl;
    }
    return 0;
}


int main_load()
{

    std::cout << "Load meshes" << std::endl;
    if (!simulation_load_fem_mesh(fem_filename.c_str()))
    {
        //return 1;
    }
    for (unsigned int a = 0; a < render_filenames.size(); ++a)
        simulation_load_render_mesh(render_filenames[a].c_str());
    if (reorder)
        simulation_reorder_fem_mesh();
    
    std::cout << "Init simulation" << std::endl;
    if (!simulation_init())
        return 1;
	return 0;
}

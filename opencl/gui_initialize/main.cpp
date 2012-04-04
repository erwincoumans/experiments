

#if defined(__APPLE__) && !defined (VMDMESA)
	#include <OpenGL/OpenGL.h>
	#include <OpenGL/gl.h>
	#include <OpenGL/glu.h>
	#include <GLUT/glut.h>
#else
#ifdef _WIN32
	#include <windows.h>
#endif
	#include <GL/gl.h>
	#include <GL/glut.h>
#endif //APPLE



#include "Gwen/Gwen.h"
#include "Gwen/Controls/Button.h"
#include "Gwen/Skins/Simple.h"
#include "Gwen/Renderers/OpenGL_DebugFont.h"
#include "Gwen/Controls/MenuStrip.h"
#include "Gwen/Controls/WindowControl.h"
#include "Gwen/Controls/ListBox.h"
#include "Gwen/Controls/VerticalSlider.h"
#include "Gwen/Controls/HorizontalSlider.h"
#include "Gwen/Controls/GroupBox.h"
#include "Gwen/Controls/CheckBox.h"
#include "Gwen/Controls/TreeControl.h"


Gwen::Renderer::OpenGL_DebugFont * pRenderer =0;
Gwen::Skin::Simple skin;
Gwen::Controls::Canvas* pCanvas =0;
class MyProfileWindow* prof = 0;

int sGlutScreenWidth = 640;
int sGlutScreenHeight = 480;
int sLastmousepos[2] = {0,0};

#include "../basic_initialize/btOpenCLUtils.h"

cl_context			g_cxMainContext;
cl_command_queue	g_cqCommandQue;



class MyProfileWindow : public Gwen::Controls::WindowControl
{
	//		Gwen::Controls::TabControl*	m_TabControl;
	Gwen::Controls::ListBox*	m_TextOutput;
	unsigned int				m_iFrames;
	float						m_fLastSecond;

	Gwen::Controls::TreeNode* m_node;
	Gwen::Controls::TreeControl* m_ctrl;

protected:

	void onButtonA( Gwen::Controls::Base* pControl )
	{
	//		OpenTissue::glut::toggleIdle();
	}

	void SliderMoved(Gwen::Controls::Base* pControl )
	{
		Gwen::Controls::Slider* pSlider = (Gwen::Controls::Slider*)pControl;
		//this->m_app->scaleYoungModulus(pSlider->GetValue());
		//	printf("Slider Value: %.2f", pSlider->GetValue() );
	}


	void	OnCheckChangedStiffnessWarping (Gwen::Controls::Base* pControl)
	{
		Gwen::Controls::CheckBox* labeled = (Gwen::Controls::CheckBox* )pControl;
		bool checked = labeled->IsChecked();
		//m_app->m_stiffness_warp_on  = checked;
	}
public:

	void MenuItemSelect(Gwen::Controls::Base* pControl)
	{
		if (Hidden())
		{
			SetHidden(false);
		} else
		{
			SetHidden(true);
		}
	}


	MyProfileWindow (	Gwen::Controls::Base* pParent)
		: Gwen::Controls::WindowControl( pParent )
	{
		
		SetTitle( L"OpenCL info" );

		SetSize( 550, 350 );
		this->SetPos(0,40);

//		this->Dock( Gwen::Pos::Bottom);

		

		{
			m_ctrl = new Gwen::Controls::TreeControl( this );
	
			int numPlatforms = btOpenCLUtils::getNumPlatforms();
			for (int i=0;i<numPlatforms;i++)
			{
				cl_platform_id platform = btOpenCLUtils::getPlatform(i);
				btOpenCLPlatformInfo platformInfo;
				btOpenCLUtils::getPlatformInfo(platform, &platformInfo);
				cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
				cl_int errNum;
				cl_context context = btOpenCLUtils::createContextFromPlatform(platform,deviceType,&errNum);
				
				
				Gwen::UnicodeString strIn = Gwen::Utility::StringToUnicode(platformInfo.m_platformName);
				Gwen::UnicodeString txt = Gwen::Utility::Format( L"Platform %d (",i)+strIn + Gwen::Utility::Format(L")");
				
				m_node = m_ctrl->AddNode(txt);
				int numDev = btOpenCLUtils::getNumDevices(context);
				for (int j=0;j<numDev;j++)
				{
					Gwen::UnicodeString txt = Gwen::Utility::Format( L"Device %d", j);
					Gwen::Controls::TreeNode* deviceNode = m_node->AddNode( txt );

					cl_device_id device = btOpenCLUtils::getDevice(context,j);
					btOpenCLDeviceInfo info;
					btOpenCLUtils::getDeviceInfo(device,&info);

					Gwen::Controls::TreeNode* node;
					Gwen::UnicodeString strIn;


					switch (info.m_deviceType)
					{
					case CL_DEVICE_TYPE_CPU:
						{
							txt = Gwen::Utility::Format( L"CL_DEVICE_TYPE_CPU");
							node = deviceNode->AddNode( txt );
							break;
						}
					case CL_DEVICE_TYPE_GPU:
						{
							txt = Gwen::Utility::Format( L"CL_DEVICE_TYPE_GPU");
							node = deviceNode->AddNode( txt );
							break;
						}
					case CL_DEVICE_TYPE_ACCELERATOR:
						{
							txt = Gwen::Utility::Format( L"CL_DEVICE_TYPE_ACCELERATOR");
							node = deviceNode->AddNode( txt );
							break;
						}
						

					default:
						{
							txt = Gwen::Utility::Format( L"Unknown device type");
							node = deviceNode->AddNode( txt );
						}
					}

					strIn = Gwen::Utility::StringToUnicode(info.m_deviceName);
					txt = Gwen::Utility::Format( L"CL_DEVICE_NAME: \t\t\t\t\t")+strIn;
					node = deviceNode->AddNode( txt );

					strIn = Gwen::Utility::StringToUnicode(info.m_deviceVendor);
					txt = Gwen::Utility::Format( L"CL_DEVICE_VENDOR: \t\t\t\t")+strIn;
					node = deviceNode->AddNode( txt );
					
					strIn = Gwen::Utility::StringToUnicode(info.m_driverVersion);
					txt = Gwen::Utility::Format( L"CL_DRIVER_VERSION: \t\t\t\t")+strIn;
					node = deviceNode->AddNode( txt );
					
					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u",info.m_computeUnits);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u",info.m_workitemDims);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u",info.m_workItemSize[0], info.m_workItemSize[1], info.m_workItemSize[2]);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u",info.m_workgroupSize);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz",info.m_clockFrequency);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_ADDRESS_BITS:\t\t%u",info.m_addressBits);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte",(unsigned int)(info.m_maxMemAllocSize/ (1024 * 1024)));
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte",(unsigned int)(info.m_globalMemSize/ (1024 * 1024)));
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s",info.m_errorCorrectionSupport== CL_TRUE ? L"yes" : L"no");
					node = deviceNode->AddNode( txt );


					txt = Gwen::Utility::Format( L"CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s",info.m_localMemType == 1 ? L"local" : L"global");
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte",(unsigned int)(info.m_localMemSize / 1024));
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte",(unsigned int)(info.m_constantBufferSize / 1024));
					node = deviceNode->AddNode( txt );

					if( info.m_queueProperties  & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
					{
						txt = Gwen::Utility::Format( L"CL_DEVICE_QUEUE_PROPERTIES:\t\tCL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
						node = deviceNode->AddNode( txt );
					}
					if( info.m_queueProperties & CL_QUEUE_PROFILING_ENABLE )
					{
						txt = Gwen::Utility::Format( L"CL_DEVICE_QUEUE_PROPERTIES:\t\tCL_QUEUE_PROFILING_ENABLE");
						node = deviceNode->AddNode( txt );
					}

					txt = Gwen::Utility::Format( L"CL_DEVICE_IMAGE_SUPPORT:\t\t%u",info.m_imageSupport);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u",info.m_maxReadImageArgs);
					node = deviceNode->AddNode( txt );


					txt = Gwen::Utility::Format( L"CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u",info.m_maxWriteImageArgs);
					node = deviceNode->AddNode( txt );

					txt = Gwen::Utility::Format( L"CL_DEVICE_EXTENSIONS");
					Gwen::Controls::TreeNode* extensionNode = deviceNode->AddNode( txt );

					if (info.m_deviceExtensions)
					{
						Gwen::Utility::Strings::List outbits;

						Gwen::String str(info.m_deviceExtensions);

						Gwen::String sep(" ");
						Gwen::Utility::Strings::Split(str,sep, outbits);
						
						

						for (int k=0;k<outbits.size();k++)
						{
							if (outbits.at(k).size())
							{
								txt = Gwen::Utility::StringToUnicode(outbits.at(k));
								node = extensionNode->AddNode( txt );
							}
						}
					}
					
/*	
	
	
	
	printf("\n  CL_DEVICE_IMAGE <dim>"); 
	printf("\t\t\t2D_MAX_WIDTH\t %u\n", info.m_image2dMaxWidth);
	printf("\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", info.m_image2dMaxHeight);
	printf("\t\t\t\t\t3D_MAX_WIDTH\t %u\n", info.m_image3dMaxWidth);
	printf("\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", info.m_image3dMaxHeight);
	printf("\t\t\t\t\t3D_MAX_DEPTH\t %u\n", info.m_image3dMaxDepth);
	
	printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t"); 
	printf("CHAR %u, SHORT %u, INT %u,LONG %u, FLOAT %u, DOUBLE %u\n\n\n", 
		info.m_vecWidthChar, info.m_vecWidthShort, info.m_vecWidthInt, info.m_vecWidthLong,info.m_vecWidthFloat, info.m_vecWidthDouble); 
*/

				}
			}

			m_ctrl->ExpandAll();
			m_ctrl->SetBounds( this->GetInnerBounds().x,this->GetInnerBounds().y,this->GetInnerBounds().w,this->GetInnerBounds().h);
		}

	}


	
	void	UpdateText()
	{
		static bool update=true;
		m_ctrl->SetBounds(0,0,this->GetInnerBounds().w,this->GetInnerBounds().h);
	}
	
};


struct MyTestMenuBar : public Gwen::Controls::MenuStrip
{
	MyProfileWindow* m_profileWindow;

	void MenuItemSelect(Gwen::Controls::Base* pControl)
	{
	}

	MyTestMenuBar(Gwen::Controls::Base* pParent, MyProfileWindow* prof)
		:Gwen::Controls::MenuStrip(pParent),
		m_profileWindow(prof)
	{
		{
			Gwen::Controls::MenuItem* pRoot = AddItem( L"File" );
		
			pRoot = AddItem( L"View" );
			pRoot->GetMenu()->AddItem( L"Platforms",prof,(Gwen::Event::Handler::Function)&MyProfileWindow::MenuItemSelect);

		}
	}

};


void	setupGUI(int width, int height)
{
	pRenderer = new Gwen::Renderer::OpenGL_DebugFont();
	skin.SetRender( pRenderer );

	pCanvas = new Gwen::Controls::Canvas( &skin );
	pCanvas->SetSize( width,height);
	pCanvas->SetDrawBackground( false);
	pCanvas->SetBackgroundColor( Gwen::Color( 150, 170, 170, 255 ) );

	//MyWindow* window = new MyWindow(pCanvas);
	prof = new MyProfileWindow(pCanvas);
	prof->UpdateText();

	MyTestMenuBar* menubar = new MyTestMenuBar(pCanvas, prof);


	
}


static	void glutKeyboardCallback(unsigned char key, int x, int y)
{
	
}

static	void glutKeyboardUpCallback(unsigned char key, int x, int y)
{

}

static void glutSpecialKeyboardCallback(int key, int x, int y)
{

}

static void glutSpecialKeyboardUpCallback(int key, int x, int y)
{

}


static void glutReshapeCallback(int w, int h)
{
	sGlutScreenWidth = w;
	sGlutScreenHeight = h;

	glViewport(0, 0, w, h);


	if (pCanvas)
	{
		pCanvas->SetSize(w,h);
	}
}



static void glutMouseFuncCallback(int button, int state, int x, int y)
{
	if (pCanvas)
	{
		bool handled = pCanvas->InputMouseButton(button,!state);
		if (handled)
		{
			sLastmousepos[0]	=	x;
			sLastmousepos[1]	=	y;
		}
	}

}


static void	glutMotionFuncCallback(int x,int y)
{
	if (pCanvas)
	{
		bool handled = pCanvas->InputMouseMoved(x,y,sLastmousepos[0],sLastmousepos[1]);
		if (handled )
		{
			
		}
		
	}
}


static void glutDisplayCallback(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 


	if (pCanvas)
	{
		saveOpenGLState(sGlutScreenWidth,sGlutScreenHeight);
		if (prof)
			prof->UpdateText();
		pCanvas->RenderCanvas();
		restoreOpenGLState();
	}

	glFlush();
	glutSwapBuffers();
}
static void glutMoveAndDisplayCallback()
{
	glutDisplayCallback();
}

int main(int argc, char* argv[])
{
	
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(sGlutScreenWidth, sGlutScreenHeight);
	char title[1024];
	sprintf(title,"OpenCL info, build using the %s OpenCL SDK", btOpenCLUtils::getSdkVendorName());
    glutCreateWindow(title);

	glutKeyboardFunc(glutKeyboardCallback);
	glutKeyboardUpFunc(glutKeyboardUpCallback);
	glutSpecialFunc(glutSpecialKeyboardCallback);
	glutSpecialUpFunc(glutSpecialKeyboardUpCallback);

	glutReshapeFunc(glutReshapeCallback);
    //createMenu();
	glutIdleFunc(glutMoveAndDisplayCallback);
	glutMouseFunc(glutMouseFuncCallback);
	glutPassiveMotionFunc(glutMotionFuncCallback);
	glutMotionFunc(glutMotionFuncCallback);
	glutDisplayFunc( glutDisplayCallback );

	glutMoveAndDisplayCallback();

	setupGUI(sGlutScreenWidth,sGlutScreenHeight);

	glutMainLoop();
	return 0;
}
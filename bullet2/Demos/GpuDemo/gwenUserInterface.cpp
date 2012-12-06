
#include "GwenUserInterface.h"
#include "../../../rendering/rendertest/GwenOpenGL3CoreRenderer.h"
#include "../../../rendering/rendertest/GLPrimitiveRenderer.h"
#include "Gwen/Platform.h"
#include "Gwen/Controls/TreeControl.h"
#include "Gwen/Controls/RadioButtonController.h"
#include "Gwen/Controls/VerticalSlider.h"
#include "Gwen/Controls/HorizontalSlider.h"
#include "Gwen/Controls/GroupBox.h"
#include "Gwen/Controls/CheckBox.h"

#include "Gwen/Controls/ComboBox.h"
#include "Gwen/Controls/MenuStrip.h"
#include "Gwen/Controls/Property/Text.h"
#include "Gwen/Controls/SplitterBar.h"

#include "Gwen/Gwen.h"
#include "Gwen/Align.h"
#include "Gwen/Utility.h"
#include "Gwen/Controls/WindowControl.h"
#include "Gwen/Controls/TabControl.h"
#include "Gwen/Controls/ListBox.h"
#include "Gwen/Skins/Simple.h"

struct GwenInternalData
{
	struct sth_stash;
	class GwenOpenGL3CoreRenderer*	pRenderer;
	Gwen::Skin::Simple				skin;
	Gwen::Controls::Canvas*			pCanvas;
	GLPrimitiveRenderer* m_primRenderer;

};
GwenUserInterface::GwenUserInterface()
{
	m_data = new GwenInternalData();

}
		
GwenUserInterface::~GwenUserInterface()
{
	delete m_data;
}
		

struct MyTestMenuBar : public Gwen::Controls::MenuStrip
{
	


	MyTestMenuBar(Gwen::Controls::Base* pParent)
		:Gwen::Controls::MenuStrip(pParent)
	{
//		Gwen::Controls::MenuStrip* menu = new Gwen::Controls::MenuStrip( pParent );
		{
			Gwen::Controls::MenuItem* pRoot = AddItem( L"File" );
		
			pRoot = AddItem( L"View" );
//			Gwen::Event::Handler* handler =	GWEN_MCALL(&MyTestMenuBar::MenuItemSelect );
			pRoot->GetMenu()->AddItem( L"Profiler");//,,m_profileWindow,(Gwen::Event::Handler::Function)&MyProfileWindow::MenuItemSelect);

/*			pRoot->GetMenu()->AddItem( L"New", L"test16.png", GWEN_MCALL( ThisClass::MenuItemSelect ) );
			pRoot->GetMenu()->AddItem( L"Load", L"test16.png", GWEN_MCALL( ThisClass::MenuItemSelect ) );
			pRoot->GetMenu()->AddItem( L"Save", GWEN_MCALL( ThisClass::MenuItemSelect ) );
			pRoot->GetMenu()->AddItem( L"Save As..", GWEN_MCALL( ThisClass::MenuItemSelect ) );
			pRoot->GetMenu()->AddItem( L"Quit", GWEN_MCALL( ThisClass::MenuItemSelect ) );
			*/
		}
	}

};

void	GwenUserInterface::init(int width, int height,struct sth_stash* stash,float retinaScale)
{
	m_data->m_primRenderer = new GLPrimitiveRenderer(width,height);
	m_data->pRenderer = new GwenOpenGL3CoreRenderer(m_data->m_primRenderer,stash,width,height,retinaScale);

	m_data->skin.SetRender( m_data->pRenderer );

	m_data->pCanvas= new Gwen::Controls::Canvas( &m_data->skin );
	m_data->pCanvas->SetSize( width,height);
	m_data->pCanvas->SetDrawBackground( false);
	m_data->pCanvas->SetBackgroundColor( Gwen::Color( 150, 170, 170, 255 ) );

	//MyTestMenuBar* menubar = new MyTestMenuBar(m_data->pCanvas);

	/*Gwen::Controls::GroupBox* box = new Gwen::Controls::GroupBox(m_data->pCanvas);
	box->SetText("text");
	box->SetName("name");
	box->SetHeight(500);
	*/
	/*Gwen::Controls::WindowControl* windowLeft = new Gwen::Controls::WindowControl(m_data->pCanvas);
	windowLeft->Dock(Gwen::Pos::Left);
	windowLeft->SetTitle("title");

	Gwen::Controls::WindowControl* windowBottom = new Gwen::Controls::WindowControl(m_data->pCanvas);
	windowBottom->SetHeight(100);
	windowBottom->Dock(Gwen::Pos::Bottom);
	windowBottom->SetTitle("bottom");
	*/
//	Gwen::Controls::Property::Text* prop = new Gwen::Controls::Property::Text(m_data->pCanvas);
	//prop->Dock(Gwen::Pos::Bottom);
	/*Gwen::Controls::SplitterBar* split = new Gwen::Controls::SplitterBar(m_data->pCanvas);
	split->Dock(Gwen::Pos::Center);
	split->SetHeight(300);
	split->SetWidth(300);
	*/
	Gwen::Controls::TabControl* tab = new Gwen::Controls::TabControl(m_data->pCanvas);
	tab->SetHeight(300);
	tab->SetWidth(100);
	tab->Dock(Gwen::Pos::Left);
	
	Gwen::Controls::ComboBox* box = new Gwen::Controls::ComboBox(tab);

}
		
void	GwenUserInterface::draw(int width, int height)
{
	
	if (m_data->pCanvas)
	{
		m_data->pCanvas->RenderCanvas();
		//restoreOpenGLState();
	}

}

#include "gwenWindow.h"
#include "application.h"

#include "Gwen/Platform.h"
#include "Gwen/Controls/TreeControl.h"
#include "Gwen/Controls/RadioButtonController.h"
#include "Gwen/Controls/VerticalSlider.h"
#include "Gwen/Controls/HorizontalSlider.h"
#include "Gwen/Controls/GroupBox.h"
#include "Gwen/Controls/CheckBox.h"

#include "Gwen/Gwen.h"
#include "Gwen/Align.h"
#include "Gwen/Utility.h"
#include "Gwen/Controls/WindowControl.h"
#include "Gwen/Controls/TabControl.h"
#include "Gwen/Controls/ListBox.h"




Gwen::Renderer::OpenGL_DebugFont * pRenderer =0;
Gwen::Skin::Simple skin;
Gwen::Controls::Canvas* pCanvas =0;



namespace OpenTissue
{
	namespace glut
	{
		void toggleIdle();
	};
};


struct MyHander   :public Gwen::Event::Handler
{
	Application* m_app;

	MyHander  (Application* app)
		:m_app(app)
	{
	}

	void onButtonA( Gwen::Controls::Base* pControl )
	{
			OpenTissue::glut::toggleIdle();
	}

	void SliderMoved(Gwen::Controls::Base* pControl )
	{
		Gwen::Controls::Slider* pSlider = (Gwen::Controls::Slider*)pControl;
		this->m_app->scaleYoungModulus(pSlider->GetValue());
		//	printf("Slider Value: %.2f", pSlider->GetValue() );
	}


	void	OnCheckChangedStiffnessWarping (Gwen::Controls::Base* pControl)
	{
		Gwen::Controls::CheckBox* labeled = (Gwen::Controls::CheckBox* )pControl;
		bool checked = labeled->IsChecked();
		m_app->m_stiffness_warp_on  = checked;
	}

	void	OnCheckChangedCollideGroundplane (Gwen::Controls::Base* pControl)
	{
		Gwen::Controls::CheckBox* labeled = (Gwen::Controls::CheckBox* )pControl;
		bool checked = labeled->IsChecked();
		m_app->m_collideGroundPlane = checked;
	}
	void	OnCheckChangedFixNodes (Gwen::Controls::Base* pControl)
	{
		Gwen::Controls::CheckBox* labeled = (Gwen::Controls::CheckBox* )pControl;
		bool checked = labeled->IsChecked();
		m_app->m_fixNodes = checked;
	}



};


class MyWindow : public Gwen::Controls::WindowControl
{
public:

	MyWindow (	Gwen::Controls::Base* pParent, Application* app )
		: Gwen::Controls::WindowControl( pParent )
	{
		SetTitle( L"FEM Settings" );

		SetSize( 200, 300 );

		this->Dock( Gwen::Pos::Left );

		m_TextOutput = new Gwen::Controls::ListBox( this );
		m_TextOutput->Dock( Gwen::Pos::Bottom );
		m_TextOutput->SetHeight( 100 );

		MyHander* handler= new MyHander(app);

		Gwen::Controls::HorizontalSlider* pSlider;

			Gwen::Controls::GroupBox* pGroup = new Gwen::Controls::GroupBox( this );
			pGroup->SetPos(5, 5);
			pGroup->SetSize(170, 45);
//			pGroup->Dock( Gwen::Pos::Fill );
			pGroup->SetText( "Young modulus" );

			pSlider = new Gwen::Controls::HorizontalSlider( pGroup );
			pSlider->SetPos( 5, 10 );
			pSlider->SetSize( 130, 20 );
			pSlider->SetRange( 0, 1000 );
			pSlider->SetValue( 25 );
			pSlider->onValueChanged.Add( handler, &MyHander::SliderMoved);

		Gwen::Controls::CheckBoxWithLabel* labeled = new Gwen::Controls::CheckBoxWithLabel( this );
		int yPos=55;
		{
		labeled->SetPos( 10, yPos);
		labeled->Checkbox()->SetChecked(true);
		labeled->Label()->SetText( "Stiffness warping" );
		labeled->Checkbox()->onCheckChanged.Add( handler, &MyHander::OnCheckChangedStiffnessWarping );
		Gwen::Align::PlaceBelow( labeled,pGroup, 0 );
		yPos+=20;
		}

		{
		Gwen::Controls::CheckBoxWithLabel* labeled2 = new Gwen::Controls::CheckBoxWithLabel( this );
		labeled2->SetPos( 10, yPos);
		labeled2->Checkbox()->SetChecked(true);
		labeled2->Label()->SetText( "Collide groundplane" );
		labeled2->Checkbox()->onCheckChanged.Add( handler, &MyHander::OnCheckChangedCollideGroundplane );
		yPos+=20;
		}

		{
		Gwen::Controls::CheckBoxWithLabel* labeled3 = new Gwen::Controls::CheckBoxWithLabel( this );
		labeled3->SetPos( 10, yPos);
		labeled3->Checkbox()->SetChecked(app->m_fixNodes);
		labeled3->Label()->SetText( "Fix bar" );
		labeled3->Checkbox()->onCheckChanged.Add( handler, &MyHander::OnCheckChangedFixNodes );
		yPos+=20;
		}



		if (0)
		{
			Gwen::Controls::GroupBox* pGroup = new Gwen::Controls::GroupBox( this );
			pGroup->SetPos(5, 55);
			pGroup->SetSize(170, 45);
	//		pGroup->Dock( Gwen::Pos::Fill );
			pGroup->SetText( "Gravity" );

			Gwen::Controls::HorizontalSlider* pSlider = new Gwen::Controls::HorizontalSlider( pGroup);
			pSlider->SetPos( 5, 10 );
			pSlider->SetSize( 130, 20 );
			pSlider->SetRange( 0, 100 );
			pSlider->SetValue( 25 );
			pSlider->SetNotchCount( 10 );
			pSlider->SetClampToNotches( true );
			//		pSlider->onValueChanged.Add( this, &Slider::SliderMoved );
		}


		Gwen::Controls::Button* pButton = new Gwen::Controls::Button( this );
		pButton->onPress.Add(handler,&MyHander::onButtonA);

		pButton->SetBounds( 5, yPos, 170, 45);
		pButton->SetText( "Toggle simulation" );



	}

	void PrintText( const Gwen::UnicodeString& str )
	{

	}

	void Render( Gwen::Skin::Base* skin )
	{
		m_iFrames++;

		if ( m_fLastSecond < Gwen::Platform::GetTimeInSeconds() )
		{
			SetTitle( Gwen::Utility::Format( L"FEM Settings  %i fps", m_iFrames ) );

			m_fLastSecond = Gwen::Platform::GetTimeInSeconds() + 1.0f;
			m_iFrames = 0;
		}

		Gwen::Controls::WindowControl::Render( skin );

	}


private:

	//		Gwen::Controls::TabControl*	m_TabControl;
	Gwen::Controls::ListBox*	m_TextOutput;
	unsigned int				m_iFrames;
	float						m_fLastSecond;

};



void	setupGUI(Application* app, int width, int height)
{
	pRenderer = new Gwen::Renderer::OpenGL_DebugFont();
	skin.SetRender( pRenderer );

	pCanvas = new Gwen::Controls::Canvas( &skin );
	pCanvas->SetSize( width,height);
	pCanvas->SetDrawBackground( false);
	pCanvas->SetBackgroundColor( Gwen::Color( 150, 170, 170, 255 ) );


	MyWindow* window = new MyWindow(pCanvas,app);

}

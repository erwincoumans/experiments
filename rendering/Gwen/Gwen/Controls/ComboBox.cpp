/*
	GWEN
	Copyright (c) 2010 Facepunch Studios
	See license in Gwen.h
*/


#include "Gwen/Controls/ComboBox.h"
#include "Gwen/Controls/Menu.h"


using namespace Gwen;
using namespace Gwen::Controls;
using namespace Gwen::ControlsInternal;

GWEN_CONTROL_CONSTRUCTOR( ComboBox )
{
	SetSize( 100, 20 );

	m_Menu = new Menu( this );
	m_Menu->SetHidden( true );
	m_Menu->SetDisableIconMargin( true );
	m_Menu->SetTabable( false );
	m_SelectedItem = NULL;

	m_OpenButton = new ComboBoxButton( this );

	m_OpenButton->onPress.Add( this, &ComboBox::OpenButtonPressed );
	m_OpenButton->Dock( Pos::Right );
	m_OpenButton->SetMargin( Margin( 2, 2, 2, 2 ) );
	m_OpenButton->SetWidth( 16 );
	m_OpenButton->SetTabable( false );

	m_SelectedText = new Label (this );
	m_SelectedText->SetAlignment( Gwen::Pos::Left | Gwen::Pos::CenterV );
	m_SelectedText->SetText( L"" );
	m_SelectedText->SetMargin( Margin( 3, 0, 0, 0 ) );
	m_SelectedText->Dock( Pos::Fill );
	m_SelectedText->SetTabable( false );

	SetTabable( true );

}

MenuItem* ComboBox::AddItem( const UnicodeString& strLabel, const String& strName, Gwen::Event::Handler* pHandler, Gwen::Event::Handler::Function fn )
{
	MenuItem* pItem = m_Menu->AddItem( strLabel, L"", pHandler, fn );
	pItem->SetName( strName );

	pItem->onMenuItemSelected.Add( this, &ComboBox::OnItemSelected );

	//Default
	if ( m_SelectedItem == NULL )
		OnItemSelected( pItem );

	return pItem;
}

void ComboBox::Render( Skin::Base* skin )
{
	skin->DrawComboBox( this );
}

void ComboBox::OpenButtonPressed( Controls::Base* /*pControl*/ )
{
	bool bWasMenuHidden = m_Menu->Hidden();

	GetCanvas()->CloseMenus();

	if ( bWasMenuHidden )
	{
		OpenList();
	}
}

void ComboBox::ClearItems()
{
	if ( m_Menu )
	{
		m_Menu->ClearItems();
	}
}
void ComboBox::OnItemSelected( Controls::Base* pControl )
{
	//Convert selected to a menu item
	MenuItem* pItem = pControl->DynamicCastMenuItem();
	m_SelectedItem = pItem;
	m_SelectedText->SetText( m_SelectedItem->GetText() );
	m_Menu->SetHidden( true );

	onSelection.Call( this );

	Focus();
	Invalidate();
}

void ComboBox::OnLostKeyboardFocus()
{
	m_SelectedText->SetTextColor( Color( 0, 0, 0, 255 ) );
}


void ComboBox::OnKeyboardFocus()
{
	//Until we add the blue highlighting again
	m_SelectedText->SetTextColor( Color( 0, 0, 0, 255 ) );
	//m_SelectedText->SetTextColor( Color( 255, 255, 255, 255 ) );
}

Gwen::Controls::Label* ComboBox::GetSelectedItem()
{	
	return m_SelectedItem;
}

void ComboBox::OpenList()
{
	if ( !m_Menu ) return;

	m_Menu->SetParent( GetCanvas() );
	m_Menu->SetHidden( false );
	m_Menu->BringToFront();

	Gwen::Point p = LocalPosToCanvas( Gwen::Point( 0, 0 ) );

	m_Menu->SetBounds( Gwen::Rect ( p.x, p.y + Height(), Width(), m_Menu->Height()) );
}

void ComboBox::CloseList()
{

}


bool ComboBox::OnKeyUp( bool bDown )
{
	if ( bDown )
	{
		Base::List::reverse_iterator it = std::find( m_Menu->Children.rbegin(), m_Menu->Children.rend(), m_SelectedItem );
		if ( it != m_Menu->Children.rend() && ( ++it != m_Menu->Children.rend() ) )
		{
			Base* pUpElement = *it;
			OnItemSelected(pUpElement);
		}
	}
	return true;
}
bool ComboBox::OnKeyDown( bool bDown )
{
	if ( bDown )
	{
		Base::List::iterator it = std::find( m_Menu->Children.begin(), m_Menu->Children.end(), m_SelectedItem );
		if ( it != m_Menu->Children.end() && ( ++it != m_Menu->Children.end() ) )
		{
			Base* pDownElement = *it;
			OnItemSelected(pDownElement);
		}
	}
	return true;
}

void ComboBox::RenderFocus( Gwen::Skin::Base* /*skin*/ )
{
}
/*
	GWEN
	Copyright (c) 2010 Facepunch Studios
	See license in Gwen.h
*/

#pragma once
#ifndef GWEN_CONTROLS_COMBOBOX_H
#define GWEN_CONTROLS_COMBOBOX_H

#include "Gwen/Controls/Base.h"
#include "Gwen/Controls/Button.h"
#include "Gwen/Gwen.h"
#include "Gwen/Skin.h"
#include "Gwen/Controls/TextBox.h"
#include "Gwen/Controls/Menu.h"


namespace Gwen 
{
	namespace Controls
	{
		class GWEN_EXPORT ComboBoxButton : public Button
		{
			GWEN_CONTROL_INLINE( ComboBoxButton, Button ){}

			virtual void Render( Skin::Base* skin )
			{
				skin->DrawComboBoxButton( this, m_bDepressed );
			}
		};

		class GWEN_EXPORT ComboBox : public Base
		{
			public:

				GWEN_CONTROL( ComboBox, Base );

				virtual void Render( Skin::Base* skin );

				Gwen::Controls::Label* GetSelectedItem();

				virtual void OpenButtonPressed( Controls::Base* pControl );
				virtual void OnItemSelected( Controls::Base* pControl );
				virtual void OpenList();
				virtual void CloseList();

				virtual void ClearItems();

				virtual MenuItem* AddItem( const UnicodeString& strLabel, const String& strName = "", Gwen::Event::Handler* pHandler = NULL, Gwen::Event::Handler::Function fn = NULL );
				bool OnKeyUp( bool bDown );
				bool OnKeyDown( bool bDown );

				void RenderFocus( Gwen::Skin::Base* skin );
				void OnLostKeyboardFocus();
				void OnKeyboardFocus();

				Gwen::Event::Caller	onSelection;

			protected:

				Menu* m_Menu;
				MenuItem* m_SelectedItem;

				ComboBoxButton* m_OpenButton;

				Label* m_SelectedText;

		};
		
	}
}
#endif

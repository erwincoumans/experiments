
#ifndef __MY_FRAME_H
#define __MY_FRAME_H

#include "wx/wx.h"

class MyFrame : public wxFrame
{

public:
	MyFrame(wxWindow* parent, const wxString& title);

	void	OnQuit(wxCommandEvent& event);
	void	OnAbout(wxCommandEvent& event);
//	void	OnSize(wxCommandEvent& event);
//	void	OnButtonOK(wxCommandEvent& event);

private:
	DECLARE_EVENT_TABLE()

};

#endif //__MY_FRAME_H

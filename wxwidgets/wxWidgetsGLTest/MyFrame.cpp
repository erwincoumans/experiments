


//EVT_SIZE(			MyFrame::OnSize)
//EVT_MENU(wxID_OK, MyFrame::OnButtonOK)
#include "MyFrame.h"

BEGIN_EVENT_TABLE(MyFrame, wxFrame)
	EVT_MENU(wxID_ABOUT, MyFrame::OnAbout)
	EVT_MENU(wxID_EXIT, MyFrame::OnQuit)
END_EVENT_TABLE()


void MyFrame::OnAbout(wxCommandEvent& event)
{
	wxString msg;
	msg.Printf(wxT("Hello and welcome to %s"), wxVERSION_STRING);
	wxMessageBox(msg,wxT("About Minimal"), wxOK|wxICON_INFORMATION, this);
}

void MyFrame::OnQuit(wxCommandEvent& event)
{
	Close();
}

MyFrame::MyFrame(wxWindow* parent, const wxString& title)
:wxFrame(parent,wxID_ANY, title)
{
	//SetIcon(wxIcon(mondrian_xpm));
	wxMenu* fileMenu = new wxMenu;
	wxMenu* helpMenu = new wxMenu;
	helpMenu->Append(wxID_ABOUT, wxT("&About...\tF1"), wxT("Show about dialog"));
	fileMenu->Append(wxID_EXIT, wxT("E&xit\tAlt-X"), wxT("Quit this program"));
	wxMenuBar* menuBar = new wxMenuBar();
	menuBar->Append(fileMenu, wxT("&File"));
	menuBar->Append(helpMenu, wxT("&Help"));
	SetMenuBar(menuBar);
	CreateStatusBar(2);
	SetStatusText(wxT("Welcome to WxWidgets!"));
}

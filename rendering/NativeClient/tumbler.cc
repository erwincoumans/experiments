// Copyright (c) 2011 The Native Client Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tumbler.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// C headers
#include <cassert>
#include <cstdio>

// C++ headers
#include <sstream>
#include <string>


#include "cube.h"
#include "opengl_context.h"
#include "scripting_bridge.h"
#include "ppapi/cpp/rect.h"
#include "ppapi/cpp/size.h"
#include "ppapi/cpp/var.h"

extern bool simulationPaused;
extern void zoomCamera(int deltaY);
extern void	mouseMotionFunc(int x,int y);
extern void mouseFunc(int button, int state, int x, int y);
extern void createWorld();
std::string sCurString="";
extern char* theData;

extern int m_glutScreenWidth;
extern int m_glutScreenHeight;


std::string ModifierToString(uint32_t modifier) {
  std::string s;
  if (modifier & PP_INPUTEVENT_MODIFIER_SHIFTKEY) {
    s += "shift ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_CONTROLKEY) {
    s += "ctrl ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_ALTKEY) {
    s += "alt ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_METAKEY) {
    s += "meta ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_ISKEYPAD) {
    s += "keypad ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_ISAUTOREPEAT) {
    s += "autorepeat ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_LEFTBUTTONDOWN) {
    s += "left-button-down ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_MIDDLEBUTTONDOWN) {
    s += "middle-button-down ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_RIGHTBUTTONDOWN) {
    s += "right-button-down ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_CAPSLOCKKEY) {
    s += "caps-lock ";
  }
  if (modifier & PP_INPUTEVENT_MODIFIER_NUMLOCKKEY) {
    s += "num-lock ";
  }
  return s;
}

std::string MouseButtonToString(PP_InputEvent_MouseButton button) {
  switch (button) {
    case PP_INPUTEVENT_MOUSEBUTTON_NONE:
      return "None";
    case PP_INPUTEVENT_MOUSEBUTTON_LEFT:
      return "Left";
    case PP_INPUTEVENT_MOUSEBUTTON_MIDDLE:
      return "Middle";
    case PP_INPUTEVENT_MOUSEBUTTON_RIGHT:
      return "Right";
    default:
      std::ostringstream stream;
      stream << "Unrecognized ("
             << static_cast<int32_t>(button)
             << ")";
      return stream.str();
  }
}

namespace {
// This is called by the brower when the 3D context has been flushed to the
// browser window.
void MyFlushCallback(void* data, int32_t result) 
{
  static_cast<tumbler::Tumbler*>(data)->FlushCallback();
}

}  // namespace


namespace {
const size_t kQuaternionElementCount = 4;
const char* const kArrayStartCharacter = "[";
const char* const kArrayEndCharacter = "]";
const char* const kArrayDelimiter = ",";

const char* const kSetCameraOrientationMessage= "SetCameraOrientation";
const char* const kHandleInputEvent= "HandleInputEvent";


// Return the value of parameter named |param_name| from |parameters|.  If
// |param_name| doesn't exist, then return an empty string.
std::string GetParameterNamed(
    const std::string& param_name,
    const tumbler::MethodParameter& parameters) {
  tumbler::MethodParameter::const_iterator i =
      parameters.find(param_name);
  if (i == parameters.end()) {
    return "";
  }
  return i->second;
}

// Convert the JSON string |array| into a vector of floats.  |array| is
// expected to be a string bounded by '[' and ']', and a comma-delimited list
// of numbers.  Any errors result in the return of an empty array.
std::vector<float> CreateArrayFromJSON(const std::string& json_array) {
  std::vector<float> float_array;
  size_t array_start_pos = json_array.find_first_of(kArrayStartCharacter);
  size_t array_end_pos = json_array.find_last_of(kArrayEndCharacter);
  if (array_start_pos == std::string::npos ||
      array_end_pos == std::string::npos)
    return float_array;  // Malformed JSON: missing '[' or ']'.
  // Pull out the array elements.
  size_t token_pos = array_start_pos + 1;
  while (token_pos < array_end_pos) {
    float_array.push_back(strtof(json_array.data() + token_pos, NULL));
    size_t delim_pos = json_array.find_first_of(kArrayDelimiter, token_pos);
    if (delim_pos == std::string::npos)
      break;
    token_pos = delim_pos + 1;
  }
  return float_array;
}
}  // namespace

namespace tumbler {

Tumbler::Tumbler(PP_Instance instance)
    : pp::Instance(instance),
      cube_(NULL) 
{
	RequestInputEvents(PP_INPUTEVENT_CLASS_MOUSE | PP_INPUTEVENT_CLASS_WHEEL);
  RequestFilteringInputEvents(PP_INPUTEVENT_CLASS_KEYBOARD);
}

Tumbler::~Tumbler() {
  // Destroy the cube view while GL context is current.
  opengl_context_->MakeContextCurrent(this);
  delete cube_;
}

void Tumbler::FlushCallback()
{
	opengl_context_->set_flush_pending(false);
	//and do the render
	DrawSelf();
}

bool Tumbler::Init(uint32_t /* argc */,
                   const char* /* argn */[],
                   const char* /* argv */[]) {
  // Add all the methods to the scripting bridge.
  ScriptingBridge::SharedMethodCallbackExecutor set_orientation_method(
      new tumbler::MethodCallback<Tumbler>(
          this, &Tumbler::SetCameraOrientation));
  scripting_bridge_.AddMethodNamed("setCameraOrientation",
                                    set_orientation_method);
  return true;
}


void Tumbler::HandleMessage(const pp::Var& message) {
	
	
	 //PostMessage("received some data");
	  //PostMessage(message.AsString());
  
  if (!message.is_string())
    return;
  
  const std::string msg = message.AsString();
  	  
  //scripting_bridge_.InvokeMethod(message.AsString());
  
  
 	//PostMessage("Handled Javascript message");
  //PostMessage(message.AsString());
  
 	if (msg=="pauseSim")
		{
			simulationPaused = !simulationPaused;
			if (simulationPaused)
			{
				PostMessage("simulation paused");
			} else
			{
				PostMessage("simulation running");
			}
  	}
		
	if (msg=="start")
		{
			PostMessage("start send data");
			sCurString="";
		} else
		if (msg=="end")
			{
				PostMessage("end send data");
				theData = (char*)sCurString.c_str();
				
				char strmsg[1024];
				sprintf(strmsg,"String length = %d\n",sCurString.length());
				PostMessage(strmsg);
				
				createWorld();
			}		else
				{
						sCurString+=msg;
				}
			
  
}

// Handle an incoming input event by switching on type and dispatching
  // to the appropriate subtype handler.
  //
  // HandleInputEvent operates on the main Pepper thread. In large
  // real-world applications, you'll want to create a separate thread
  // that puts events in a queue and handles them independant of the main
  // thread so as not to slow down the browser. There is an additional
  // version of this example in the examples directory that demonstrates
  // this best practice.
bool Tumbler::HandleInputEvent(const pp::InputEvent& event) {
    //PostMessage(pp::Var(kHandleInputEvent));
    switch (event.GetType()) {
      case PP_INPUTEVENT_TYPE_UNDEFINED:
        break;
      case PP_INPUTEVENT_TYPE_MOUSEDOWN:
      	{
      		pp::MouseInputEvent mouse_event = pp::MouseInputEvent(event);
      			
    	    if (mouse_event.GetButton()==PP_INPUTEVENT_MOUSEBUTTON_LEFT)
        	{
        		int posX = mouse_event.GetPosition().x();
        		int posY = mouse_event.GetPosition().y();
    	    		mouseFunc(0,0,posX,posY);
    	    		
    	    		
    	    		char tmp[1024];
    	    		sprintf(tmp,"Mouse Down, posX = %d posY = %d, screenWidth %d, screenHeight %d",posX,posY, m_glutScreenWidth,m_glutScreenHeight);
    	    		PostMessage(tmp);
    	    		
  	      }
        GotMouseEvent(pp::MouseInputEvent(event), "Down");
        break;
      }
      case PP_INPUTEVENT_TYPE_MOUSEUP:
      	{
      			pp::MouseInputEvent mouse_event = pp::MouseInputEvent(event);
      			
    	    if (mouse_event.GetButton()==PP_INPUTEVENT_MOUSEBUTTON_LEFT)
        	{
    	    		mouseFunc(0,1,mouse_event.GetPosition().x(),mouse_event.GetPosition().y());
    	    		PostMessage("Mouse Up");
  	      }
        GotMouseEvent(pp::MouseInputEvent(event), "Up");
        }
        break;
      case PP_INPUTEVENT_TYPE_MOUSEMOVE:
      	{
      		pp::MouseInputEvent mouse_event = pp::MouseInputEvent(event);
      		int xPos = mouse_event.GetPosition().x();
      		int yPos = mouse_event.GetPosition().y();
      			
      			mouseMotionFunc(xPos,yPos);
      				char tmp[1024];
    	    		sprintf(tmp,"Mouse Move, posX = %d posY = %d",xPos,yPos);
    	    		PostMessage(tmp);
      			
        GotMouseEvent(pp::MouseInputEvent(event), "Move");
        break;
      }
      case PP_INPUTEVENT_TYPE_MOUSEENTER:
        GotMouseEvent(pp::MouseInputEvent(event), "Enter");
        break;
      case PP_INPUTEVENT_TYPE_MOUSELEAVE:
        GotMouseEvent(pp::MouseInputEvent(event), "Leave");
        break;
      case PP_INPUTEVENT_TYPE_WHEEL:
        GotWheelEvent(pp::WheelInputEvent(event));
        break;
      case PP_INPUTEVENT_TYPE_RAWKEYDOWN:
        GotKeyEvent(pp::KeyboardInputEvent(event), "RawKeyDown");
        break;
      case PP_INPUTEVENT_TYPE_KEYDOWN:
        GotKeyEvent(pp::KeyboardInputEvent(event), "Down");
        break;
      case PP_INPUTEVENT_TYPE_KEYUP:
        GotKeyEvent(pp::KeyboardInputEvent(event), "Up");
        break;
      case PP_INPUTEVENT_TYPE_CHAR:
        GotKeyEvent(pp::KeyboardInputEvent(event), "Character");
        break;
      case PP_INPUTEVENT_TYPE_CONTEXTMENU:
        GotKeyEvent(pp::KeyboardInputEvent(event), "Context");
        break;
      default:
        assert(false);
        return false;
    }
    return true;
  }
  
  
   void Tumbler::GotKeyEvent(const pp::KeyboardInputEvent& key_event, const std::string& kind) 
   {
    std::ostringstream stream;
    stream << pp_instance() << ":"
           << " Key event:" << kind
           << " modifier:" << ModifierToString(key_event.GetModifiers())
           << " key_code:" << key_event.GetKeyCode()
           << " time:" << key_event.GetTimeStamp()
           << " text:" << key_event.GetCharacterText().DebugString()
           << "\n";
    //PostMessage(stream.str());
  }

  void Tumbler::GotMouseEvent(const pp::MouseInputEvent& mouse_event, const std::string& kind) 
  {
    std::ostringstream stream;
    stream << pp_instance() << ":"
           << " Mouse event:" << kind
           << " modifier:" << ModifierToString(mouse_event.GetModifiers())
           << " button:" << MouseButtonToString(mouse_event.GetButton())
           << " x:" << mouse_event.GetPosition().x()
           << " y:" << mouse_event.GetPosition().y()
           << " click_count:" << mouse_event.GetClickCount()
           << " time:" << mouse_event.GetTimeStamp()
           << "\n";
    //PostMessage(stream.str());

  }

  void Tumbler::GotWheelEvent(const pp::WheelInputEvent& wheel_event) 
 	{
    std::ostringstream stream;
    stream << pp_instance() << ": Wheel event."
           << " modifier:" << ModifierToString(wheel_event.GetModifiers())
           << " deltax:" << wheel_event.GetDelta().x()
           << " deltay:" << wheel_event.GetDelta().y()
           << " wheel_ticks_x:" << wheel_event.GetTicks().x()
           << " wheel_ticks_y:"<< wheel_event.GetTicks().y()
           << " scroll_by_page: "
           << (wheel_event.GetScrollByPage() ? "true" : "false")
           << "\n";
    //PostMessage(stream.str());
    zoomCamera(-wheel_event.GetDelta().y());
  }
  
void Tumbler::DidChangeView(const pp::Rect& position, const pp::Rect& clip) {
  int cube_width = cube_ ? cube_->width() : 0;
  int cube_height = cube_ ? cube_->height() : 0;
  if (position.size().width() == cube_width &&
      position.size().height() == cube_height)
    return;  // Size didn't change, no need to update anything.

  if (opengl_context_ == NULL)
    opengl_context_.reset(new OpenGLContext(this));
  opengl_context_->InvalidateContext(this);
  opengl_context_->ResizeContext(position.size());
  if (!opengl_context_->MakeContextCurrent(this))
    return;
  if (cube_ == NULL) {
    cube_ = new Cube(opengl_context_);
    cube_->PrepareOpenGL();
  }
  cube_->Resize(position.size().width(), position.size().height());
  
  m_glutScreenWidth = position.size().width();
m_glutScreenHeight = position.size().height();
  
  DrawSelf();
}

void Tumbler::DrawSelf() {
  if (cube_ == NULL || opengl_context_ == NULL)
    return;
  opengl_context_->MakeContextCurrent(this);
  cube_->Draw();
  int i=0;
  opengl_context_->FlushContext(MyFlushCallback, this);
}



void Tumbler::SetCameraOrientation(
    const tumbler::ScriptingBridge& bridge,
    const tumbler::MethodParameter& parameters) {
  // |parameters| is expected to contain one object named "orientation", whose
  // value is a JSON string that represents an array of four floats.
  if (parameters.size() != 1 || cube_ == NULL)
    return;
  std::string orientation_desc = GetParameterNamed("orientation", parameters);
  std::vector<float> orientation = CreateArrayFromJSON(orientation_desc);
  if (orientation.size() != kQuaternionElementCount) {
    return;
  }
  cube_->SetOrientation(orientation);
  DrawSelf();
  //PostMessage(kSetCameraOrientationMessage);

}
}  // namespace tumbler
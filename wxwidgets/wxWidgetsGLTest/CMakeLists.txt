set(NS_SRC   
	main.cpp
	MyFrame.cpp
)

set(NS_HDR
	MyFrame.h
)

if (WIN32)
	set(PLAT_RESOURCE Win32/application.rc)
endif()

set(RESOURCE )
set_source_files_properties(${RESOURCE} PROPERTIES HEADER_FILE_ONLY 1)
source_group(Content\\Resources FILES ${PLAT_RESOURCE} ${RESOURCE})


set(ALL
    ${NS_SRC} 
    ${NS_HDR}
    ${RESOURCE}
    ${PLAT_RESOURCE}
)

include_directories(. )

set(CMAKE_MODULE_PATH    ${BULLET_PHYSICS_SOURCE_DIR}/Extras/wxWidgets-2.9.0/wxCMake  )
set(wxWidgets_SOURCE_DIR            ${BULLET_PHYSICS_SOURCE_DIR}/Extras/wxWidgets-2.9.0)
include(System)

option(wxWidgets_DEBUG              "Enable / Disable debug mode (__WXDEBUG__)"                 OFF)
option(wxWidgets_PRECOMP            "Enable / Disable pre compiled headers under __WXMSW__"     ON)
option(wxWidgets_GROUP_HEADERS      "Sort headers into groups for the current IDE "             OFF)

setup_wx()

if (WIN32)
    add_executable(AppWxWidgetsGLTest WIN32 ${ALL})
else()
    add_executable(AppWxWidgetsGLTest ${ALL})
endif()

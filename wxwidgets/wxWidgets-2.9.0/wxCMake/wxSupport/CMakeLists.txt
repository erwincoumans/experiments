make_source(wxSOURCE ${wxSUPPORT_SOURCE})
setup_wx_support(wxSupport)
ADD_DEFINITIONS(-DNOPCH)
add_library(wxSupport ${wxSOURCE})

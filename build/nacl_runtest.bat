cd nacl
cd nginx-1.1.2
start nginx.exe 
cd ..

rem assume an appropriate Chromium build is copied in the right location
rem start chrome-win32\chrome.exe  --enable-accelerated-plugins http://localhost/index.html

echo Press a key to stop
pause
cd nginx-1.1.2
start nginx.exe -s stop
cd ..
cd ..
pause
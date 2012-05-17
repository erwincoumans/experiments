CPP = g++
CPPFLAGS = -Wall -pedantic -g -O2 -DNDEBUG
LDFLAGS = 

all: libexact-ccd.a 

libexact-ccd.a: expansion.o interval.o rootparitycollisiontest.o 
	ar r $@ expansion.o interval.o rootparitycollisiontest.o;
	ranlib $@

expansion.o: expansion.cpp expansion.h
	$(CPP) $(CPPFLAGS) -o $@ -c expansion.cpp

interval.o: interval.cpp interval.h
	$(CPP) $(CPPFLAGS) -o $@ -c interval.cpp

rootparitycollisiontest.o: rootparitycollisiontest.cpp expansion.h interval.h vec.h
	$(CPP) $(CPPFLAGS) -o $@ -c rootparitycollisiontest.cpp

clean:
	-rm libexact-ccd.a expansion.o interval.o rootparitycollisiontest.o

#include "tinyxml.h"
#include <time.h>
#include <fstream>

using namespace std;

int main( int argc, char* argv[] )
{
	if ( argc < 2 )
	{
		printf( "Loads a file, then writes it back out.\n" );
		printf( "Usage: echo inputfile\n" );
		return 1;
	}

	clock_t start = clock();

	TiXmlDocument doc( argv[1] );
	doc.LoadFile();
	if ( doc.Error() )
	{
		printf( "Error loading document.\n" );
		printf( "Error id=%d desc='%s' row=%d col=%d\n",
				 doc.ErrorId(), doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol() );
		return 2;
	}
	
	printf( "Load '%s' successful.\n", doc.Value() );

/*
#ifdef TIXML_USE_STL	
	printf( "STL mode on.\n" );
	doc.SaveFile( "echotest.stl.xml" );
#else
	printf( "STL mode OFF.\n" );
#endif
	doc.SaveFile( "echotest.xml" );

	doc.Print( stdout, 0 );
*/
	TiXmlPrinter printer;
	doc.Accept( &printer );
	
	clock_t end = clock();
	printf( "Clocks: %d\n", (int)(end-start) );
	return 0;
}


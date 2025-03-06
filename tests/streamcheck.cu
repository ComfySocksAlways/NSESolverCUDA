#include <stdio.h>

int main( void ) {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice( &whichDevice );
    cudaGetDeviceProperties( &prop, whichDevice ) ;
    if(!prop.deviceOverlap){
            printf( "Device will not handle overlaps, so no speed up from streams\n" );
    }else{
        printf( "Device will handle overlaps\n" );
    }
return 0;
}
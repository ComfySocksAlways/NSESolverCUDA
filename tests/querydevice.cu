#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

int main(int argc, char *argv[])
{
    cudaDeviceProp prop;
    cudaError_t status;
    int device_count;
    int device_index = 0;
    if (argc > 1) {
        device_index = atoi(argv[1]);
    }

    status = cudaGetDeviceCount(&device_count);
    printf("Device count=%d\n", device_count);
    if (status != cudaSuccess) {
        fprintf(stderr,"cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    if (device_index >= device_count) {
        fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", device_index, device_count);
        return -1;
    }
    status = cudaGetDeviceProperties(&prop, device_index);
    if (status != cudaSuccess) {
        fprintf(stderr,"cudaGetDeviceProperties() for device device_index failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    int v = prop.major * 10 + prop.minor;
    printf("Compute Capability: %d\n", v);

    int threads_per_block = prop.warpSize;
    size_t sharedMemorySize = threads_per_block * sizeof(double);

     if (sharedMemorySize <= prop.sharedMemPerBlock){
         printf("Enough shared memory\nRequired: %ld \nAvailable: %ld\n", sharedMemorySize, prop.sharedMemPerBlock);
     }
     else{
         printf("Not Enough shared memory\nRequired: %ld \nAvailable: %ld\n", sharedMemorySize, prop.sharedMemPerBlock);
     }
}
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    cudaDeviceProp device{};
    const cudaError_t status = cudaGetDeviceProperties(&device, 0);
    if (status != cudaSuccess) {
        std::fprintf(stderr,"CUDA device query failed: %s\n",cudaGetErrorString(status));
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1],"--arch") == 0) {
        std::printf("%d%d\n",device.major,device.minor);
    } else {
        int driver = 0, runtime = 0;
        if (cudaDriverGetVersion(&driver) != cudaSuccess || cudaRuntimeGetVersion(&runtime) != cudaSuccess) return 1;
        std::printf("name=%s\narchitecture=%d%d\nmemory_bytes=%zu\ndriver=%d\nruntime=%d\n",
            device.name,device.major,device.minor,size_t(device.totalGlobalMem),driver,runtime);
    }
    return 0;
}

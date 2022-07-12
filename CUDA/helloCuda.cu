#include <stdio.h>

__global__ void greeting(){
        printf("Hello, World\n");
}


int main(){
        greeting<<<2, 3>>>();

        cudaDeviceSynchronize();
}
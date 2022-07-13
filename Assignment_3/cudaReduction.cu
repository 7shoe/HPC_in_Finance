#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include <chrono>       /* measure time */

static const int bSize = 1024; // multipe of 2

__global__ void gpuAdd(float *a, float *b, float *c, int N){

        int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;

        for (int idx=threadId; idx<N; idx += stride){
                c[idx] = a[idx] + b[idx];
        }

        //std::cout << "blockDim.x: " << blockDim.x << std::endl;
        //std::cout << "gridDim.x: " << gridDim.x << std::endl;
}

__global__ void gpuReduction(const float *d_a, float *sum, int N) {
        int idx = threadIdx.x;
        float s = 0.0f;

        for (int i = idx; i < N; i += bSize){
                s += d_a[i];
        }
        __shared__ float dyadic[bSize];
        dyadic[idx] = s;
        __syncthreads();

        for (int i = bSize/2; i>0; i/=2) {
                //uniform
                if (idx<i){
                        dyadic[idx] += dyadic[i+idx];
                }
                __syncthreads();
        }
        if(idx == 0){
                *sum=dyadic[0];
        }
}

int main(){
        // declare host memory and put data in
        int N = 50500;

        float host_sum = 0.0f;
        float* h_sum = &host_sum;
        float* h_a, *h_b, *h_c;

        // allocate memory
        h_a = (float *)malloc(N*sizeof(float));
        h_b = (float *)malloc(N*sizeof(float));
        h_c = (float *)malloc(N*sizeof(float));

        h_sum = (float *)malloc(sizeof(float));

        // init data
        for (int i=0; i<N; ++i){
                h_a[i] = (float)N-i-1;
                h_b[i] = (float)i-N+i+2;
        }

        // declare device memory
        float* d_a, *d_b, *d_c, *d_sum;

        cudaMalloc((float**)&d_a, N*sizeof(float));
        cudaMalloc((float**)&d_b, N*sizeof(float));
        cudaMalloc((float**)&d_c, N*sizeof(float));

        cudaMalloc((float**)&d_sum, sizeof(float));

        // start the clock
        std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();

        // copy data to device
        cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

        // use device function
        gpuAdd<<<16, 64>>>(d_a, d_b, d_c, N);

        // perform reduction here
        gpuReduction<<<1, bSize>>>(d_c, d_sum, N);

        cudaDeviceSynchronize();

        // copy sum
        cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

        // measure time 
        std::chrono::high_resolution_clock::time_point t12 = std::chrono::high_resolution_clock::now();

        // free up device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // free host memory
        free(h_a);
        free(h_b);
        free(h_c);

        //output
        std::cout << "sum: " << *h_sum << std::endl;

        // Time measurements
        std::cout << "cudaMemcpy: host->device: " << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count() << " ms\n" << std::endl;
}
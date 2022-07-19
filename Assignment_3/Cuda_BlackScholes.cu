



#include <iostream>
#include <math.h>
#include <chrono>

__global__ void BlackScholesKernel(float* C, float* S, float K, float r, float T, float v){

        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        float d_1 = (std::log(S[idx]/K) + (r + 0.5*v*v)*T) / (v * std::sqrt(T));
        float d_2 = d_1 - v * sqrtf(T);

        C[idx]= normcdff(d_1)*S[idx] - normcdff(d_2) * K * std::exp(-r*T);
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

        // start the clock
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // parameters
        const int N     = 1000000;
        const int bSize = 1000;
        float K = 100.0;
        float r = 0.03;
        float T = 1.0;
        float v = 0.3;

        // host memory
        float *h_C, *h_S;
        h_C = (float *)malloc(N*sizeof(float));
        h_S = (float *)malloc(N*sizeof(float));

        // initialize data
        for (int i=0; i<N; ++i){
                h_S[i] = 100.0;
                h_C[i] = 0.0;
        }

        // allocate memory on the device
        float* d_C, *d_S, *d_sum;
        cudaMalloc((float**)&d_C, N*sizeof(float));
        cudaMalloc((float**)&d_S, N*sizeof(float));
        cudaMalloc((float**)&d_sum, sizeof(float));

        // copy data from host to memory
        cudaMemcpy(d_S, h_S, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, N*sizeof(float), cudaMemcpyHostToDevice);

        // run kernel
        BlackScholesKernel<<<BSIZE, N / BSIZE>>>(d_C, d_S, K, r, T, v);

        // synchronize
        cudaDeviceSynchronize();

        // copy data from device to host
        cudaMemcpy(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_S, d_S, N*sizeof(float), cudaMemcpyDeviceToHost);

        // stop the clock
        //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        // perform reduction here
        gpuReduction<<<1, bSize>>>(d_C, d_sum, N);

        // stop the clock
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        std::cout << "C: " << h_C[5] << std::endl;

                // output time
        std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n" << std::endl;

        // free
        cudaFree(d_C);
        cudaFree(d_S);

        free(h_C);
        free(h_S);


        return 0;
}
                                                                                                                                         105,1         
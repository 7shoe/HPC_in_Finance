#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>
#include <chrono>

#define S1      100.0f
#define S2      110.0f
#define S3      90.0f

typedef std::numeric_limits<double> DblLim;

// tuned w.r.t. Midway's GPU  (Quadro RTX 6000) 
const int bSize     = 128;    // multipe of 2 for dyadic reduction from sum to mean
const int WARP_SIZE = 32;     // Warp size (max. threads, see: CL_NV_DEVICE_WARP_SIZE) 
const int N_BLOCKS  = 4262;   // Number of total cuda cores on my GPU
const int N_ITER    = 22;     // Mutiple of 2! 

// Monte Carlo Kernel
__global__ void monte_carlo(float3 *c_prices, float mu, float sigma, float K) {
        // Define some shared memory: all threads in this block
        __shared__ float3 call_prices[WARP_SIZE];
        float2 s1, s2, s3;

        // unique ID of the threaid
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // most efficient rnd generator (given log-normal dist)
        curandStatePhilox4_32_10_t rng;
        curand_init(clock64(), idx, 0, &rng);

        //init
        call_prices[threadIdx.x].x = 0;
        call_prices[threadIdx.x].y = 0;
        call_prices[threadIdx.x].z = 0;

        // generate random numbers (two at a time via efficient curand_log_normal2)
        for (int i = 0; i < N_ITER / 2; i++) {
                s1 = curand_log_normal2(&rng, mu, sigma);
                s2 = curand_log_normal2(&rng, mu, sigma);
                s3 = curand_log_normal2(&rng, mu, sigma);

                // transform to payoff
                s1.x = s1.x * S1 - K;
                s1.y = s1.y * S1 - K;
                s2.x = s2.x * S2 - K;
                s2.y = s2.y * S2 - K;
                s3.x = s3.x * S3 - K;
                s3.y = s3.y * S3 - K;

                // add (if positive payoff
                call_prices[threadIdx.x].x += (s1.x>0 ? s1.x : 0.0) + (s1.y>0 ? s1.y : 0.0); // S=100
                call_prices[threadIdx.x].y += (s2.x>0 ? s2.x : 0.0) + (s2.y>0 ? s2.y : 0.0); // S=110
                call_prices[threadIdx.x].z += (s3.x>0 ? s3.x : 0.0) + (s3.y>0 ? s3.y : 0.0); // S=90
        }


        // aggregate block-wise
        if(threadIdx.x == 0){
// Accumulate results
                c_prices[blockIdx.x].x = 0;
                c_prices[blockIdx.x].y = 0;
                c_prices[blockIdx.x].z = 0;

                // aggregate
                for (int i = 0; i < WARP_SIZE; i++) {
                        c_prices[blockIdx.x].x += call_prices[i].x;
                        c_prices[blockIdx.x].y += call_prices[i].y;
                        c_prices[blockIdx.x].z += call_prices[i].z;
                }
        }
}

__global__ void gpuReduction(const float3 *d_a, float *dC1, float *dC2, float *dC3, int N, int N_ITER, int WARP_SIZE, float r, float t){

        int idx = threadIdx.x;

        // discount factor: temporal & to convert sum to mean
        float disc = __expf(-r*t) / (float)(N_ITER * N * WARP_SIZE);

        float3 s;
        s.x = 0.0f;
        s.y = 0.0f;
        s.z = 0.0f;

        for (int i = idx; i < N; i += bSize){
                s.x += d_a[i].x;
                s.y += d_a[i].y;
                s.z += d_a[i].z;
        }
        __shared__ float3 dyadic[bSize];
        dyadic[idx].x = s.x;
        dyadic[idx].y = s.y;
        dyadic[idx].z = s.z;

        __syncthreads();

        for (unsigned int i = bSize/2; i>0; i/=2) {
                //uniform
                if (idx<i){
                        dyadic[idx].x += dyadic[i+idx].x;
                        dyadic[idx].y += dyadic[i+idx].y;
                        dyadic[idx].z += dyadic[i+idx].z;
                }
                __syncthreads();
        }
        if(idx == 0){
                *dC1 = dyadic[0].x * disc;
                *dC2 = dyadic[0].y * disc;
                *dC3 = dyadic[0].z * disc;
        }
}

int main(){
        // start the clock
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // input parameters 
        float r = 0.03;
        float t = 1.0;
        float K = 100.0;
        float v = 0.3;

        // allocate host and device memory to store the counters
        float3 *dOut;
        float *dC1, *dC2, *dC3;

        // device memory (leverage GPU-specific datatype)
        cudaMalloc((float3 **)&dOut, N_BLOCKS * sizeof(float3));

        // joint host/device memory (scalars only)
        cudaMallocManaged(&dC1, sizeof(float));
        cudaMallocManaged(&dC2, sizeof(float));
        cudaMallocManaged(&dC3, sizeof(float));

        // kernel : payout simulation
        monte_carlo<<<N_BLOCKS, WARP_SIZE>>>(dOut, (r - 0.5*v*v)*t, v, K);

        // synchronize
        cudaDeviceSynchronize();

        gpuReduction<<<1, bSize>>>(dOut, dC1, dC2, dC3, N_BLOCKS, N_ITER, WARP_SIZE, r, t);

        cudaDeviceSynchronize();
        cudaFree(dOut);

        // stop the clock
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout << "\nPart 1:" << std::endl;
        std::cout << "Wall time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n" << std::endl;

        // re-run & show option prices
        std::cout << "\nPart 2:" << std::endl;

        cudaMalloc((float3 **)&dOut, N_BLOCKS * sizeof(float3));
        monte_carlo<<<N_BLOCKS, WARP_SIZE>>>(dOut, (r - 0.5*v*v)*t, v, K);
        cudaDeviceSynchronize();

        gpuReduction<<<1, bSize>>>(dOut, dC1, dC2, dC3, N_BLOCKS, N_ITER, WARP_SIZE, r, t);
        cudaDeviceSynchronize();

        // option call prices
        std::cout << "C(S=100) = "  << *dC1  << std::endl;
        std::cout << "C(S=110) = "  << *dC2  << std::endl;
        std::cout << "C(S=90)  = "  << *dC3  << std::endl;

        // samples 
        int N_SAMPLES = N_ITER * N_BLOCKS * WARP_SIZE;
        std::cout << "\nNo. of samples: " << std::lround((float)N_SAMPLES / 1e6) << " mio (i.e. N=" << std::lround((1.0/3.0)*(float)N_SAMPLES / 1e6) << " mio)."  << std::endl;
        return 0;
}
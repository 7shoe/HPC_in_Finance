#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>

typedef double Price;
typedef std::numeric_limits<double> DblLim;

const unsigned long WARP_SIZE   = 32;       // Warp size
const unsigned long NBLOCKS     = 1000;      // Number of total cuda cores on my GPU
const unsigned long N           = 1000; // Number of points to generate (each thread)

__global__ void BlackScholes(double *Calls){

        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        float mu    = -0.015;
        float sigma = 0.3;
        float S = 110.0;
        float K = 100.0;

        curandState_t curState;
        curand_init(clock64(), idx, 0, &curState);

        //int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float x = curand_log_normal(&curState, mu, sigma) * S - K;
        x = x > 0.0 ? x : 0.0f;

        Calls[threadIdx.x] = x;
}

__global__ void Call_MC(Price *means, const double& S, const double& K){
        // define some shared memory: all threads in this block
        __shared__ Price call_prices[WARP_SIZE];

        // unique ID of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // logNormal Parameter
        float mu    = -0.015;
        float sigma = 0.3;

        // Initialize RNG
        curandState_t curState;
        curand_init(clock64(), idx, 0, &curState);

        // Initialize the counter
        call_prices[threadIdx.x] = 0.0;

        // Computation loop
        for (int i = 0; i < N; i++) {
                //float x = curand_uniform(&curState);                                 // Random x position in [0,1]
                float x = curand_log_normal(&curState, mu, sigma);              // log-normally distributed random number
                //counter[threadIdx.x] += 1 - int(x * x + y * y);   // Hit test
                x = x*S - K;
                call_prices[threadIdx.x] += x;
                //call_prices[threadIdx.x] += x / (double)N;
                //std::cout << "x: " << x << std::endl;
        }
         //__syncthreads();

        // The first thread in *every block* should sum the results
        if (threadIdx.x == 0){
                // Reset count for this block
                means[blockIdx.x] = 0.0;
                // Accumulate results
                for (int i = 0; i < WARP_SIZE; i++){
                        means[blockIdx.x] += call_prices[i] / (double) WARP_SIZE;
                }
        }
}

int main() {

        //int numDev;
        //cudaGetDeviceCount(&numDev);

        //if(numDev < 1){
        //      std::cout << "CUDA device missing! Do you need to use optirun?\n";
        //      return 1;
        //}

        //std::cout << "Starting simulation with " << NBLOCKS << " blocks, " << WARP_SIZE << " threads, and " << N << " iterations\n";

        // Allocate host and device memory to store the counters
        //Price *h_mean, *d_mean;
        //h_mean = new Price[NBLOCKS]; 

        double *d_prices, *h_prices; //, *stock_prices, *strike_prices;


        // Host memory
        cudaMalloc(&d_prices, sizeof(double) * 10000); // Device memory

        // Launch kernel
        BlackScholes<<<100, 100>>>(d_prices);

        // Copy back memory used on device and free
        cudaMemcpy(h_prices, d_prices, sizeof(double) * N, cudaMemcpyDeviceToHost);
        cudaFree(d_prices);

        // Compute total hits
        double totalMean = 0.0;
        for (int i = 0; i < N; i++){
                totalMean += h_prices[i] / (double)N;
        }

        // DEBUG
        //std::cout << "DEBUG, h_mean[66]] " << h_mean[66] << std::endl; 

        //int total_sims = NBLOCKS * N * WARP_SIZE;

        std::cout << "C_T: " << totalMean * std::exp(-1.0 * 0.03) << std::endl;

        // Set maximum precision for decimal printing
        //std::cout.precision(DblLim::max_digits10);
        //std::cout << "C = " << totalMean << std::endl;

        return 0;

}


# 0. CUDA on Midway

In a nutshell. To run code on any device, run:
```
module load cuda/11.5

sinteractive --partition=gpu --gres=gpu:1 --time=0:30:00 --account=finm32950 
```

More details:
Copy and unzip `.tar` file. Activate CUDA module.
```
cp /project/finm32950/chanaka/L5Demo.tar L5Demo.tar

module avail cuda
> cuda/10.2  cuda/11.2  cuda/11.3  cuda/11.5

module load cuda/11.5

sinteractive --partition=gpu --gres=gpu:1 --time=0:30:00 --account=finm32950

nvcc hello.cu -o hello
```

Switch back to C++ compiler if you have to
```
module avail
module list
module load intel/2022.0
module unload intel/2022.0
```

# 1. CUDA Essentials

### MVP
A function annotated with `__global__` can be devised from the host (i.e. CPU) to be run on the device (i.e. GPU).
Note, the CUDA kernel is a sequential program! This function that is supposed to be run on the device is called **kernel**. In the example below, `fun()` is therefore a kernel. 
The kernel is executed in parallel by CUDA threads. 
```
__global__ void fun(){
    // do work()
}
```
When launching a kernel, we must provide an **execution configuration**, which is done by using the <<< ... >>> syntax.
The execution configuration specifies the number of threads. In particular, `<<<k, n>>>` requests `k` blocks with `n` GPU threads per block. In turn, the total number of threads is `n*k`. 
```
int main(){
    fun<<<1, 1000 >>>
}

```
Finally, function `cudaDeviceSynchronize();`

### Keywords
- `__global__`: callable from host or device, run on device; must return `void`. 
- `__device__`: callable from device only *and* run on device
- `__host__`: 

## Data Parallel Problems 
CUDA is a formidable technology for data parallelism
- data parallelism: execute the **same operation** on **different data**
- [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) (single instruction, multiple threads): Noticably, all "threads" are executed in lock-step. Popular on general-purpose computing on graphics processing units (GPGPU). 

The major drawback is the transfer of data in between memories of host and device. This tens to be expensive. 

### Interlude: `malloc` 
A construct of c++ that allocates size bytes of uninitialized storage. If allocation succeeds, it returns a pointer that is suitably aligned for any object type.

Note that `malloc` is thread-safe: it behaves as though only accessing the memory locations visible through its argument, and not any static storage.
```
cont int N = 10000;
int numBytes = N * sizeof(double);
double* a = (double *)malloc(numBytes));  // allocates enough for an array of 10k double vals

if(a){
    for(int i=0; i<N; ++i){
        p1[n] = n*n;
    }     
}

free(a);
```

### Grids, Blocks, Thread Hierarchy
As easy to read [summary](https://eximia.co/understanding-the-basics-of-cuda-thread-hierarchies/). 
The host launches the kernel. All threads in a kernel launch are called *grid*. A *block* is a group of threads. 
CUDA runtime assigns each thread `2` variables, e.g. the following code requests 16 threads for each of the 4 blocks:
```
myKernel<<<4, 16>>>
```
Or,
```
dim3 dimGrid(2, 2, 1);
dim3 dimBlock(4,2,2);
```
The number of threads per block is constant for a given CPU. The maximum value is `1024`. As a rule-of-thumb, the block size is chosen to be a multiple of `32`. The maximum number of blocks, on the other hand, depends on the *compute capability*. 

# Thread/Block IDs
`threadIdx.x` and `blockIdx` can be used to define a unique thread ID with the following scheme:
```
int threadID = threadIdx.x + blockIdx.x * blockDim.x;
```
e.g.
```
__global__ void add(float* a, floiat* b, float* c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```
Keyword `cudaMemcpyDeviceToHost`. 

## Grid-Stride Range_based for Loop
If `N` is bigger than the number of available threads, block several operations into one via *grid-stride loop*. In our example: 
```
for(int idx=threadId; i<N; idx+=strid){
    c[idx] = a[idx] + b[idx];
}
```
It handles `N>T`, scales well for different sizes of `N`, and increases efficiency as each thread is computed.


## Naming Convention
`d_a` for variables referring ot objects run on the device while `h_a` will be run on the host. In addition, CUDA provides the C++-alike `cudaMalloc` to store data. The data can be transferred from the host to the device and vice versa, e.g. through the `cudaMemcpy()` function.

```
cudaMalloc((float**)&d_a, N*sizeof(float)); 
cudaMalloc((float**)&d_b, N*sizeof(float)); 
cudaMalloc((float**)&d_c, N*sizeof(float)); 

cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice)  // copy data from host -> device
cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice)  // copy data from host -> device

...

cudaMemcpy(h_a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost)  // copy back from device ->host
```


## Further Reading on CUDA Fundamentals
The official [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html). The documentation is available as a 454-page strong [PDF](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf).

An interesting [blog article](https://developer.nvidia.com/blog/power-cpp11-cuda-7/) by Mark Harris on CUDA's capability in  light of C++ 11 features.
In addition, another [article](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) of the same author on how to reduce data transfer.

# 2. Examples in Finance
A throwback to the first problem in this class. Pricing many options.
Now, define a kernel that is the Black Scholes pricer.

## 2.1 Black Scholes Pricing
```
__global__ void BlackScholesKernel(
    float *S, float *K, float *T, 
    float *r, float* v, 
    float* C, Float *P)
```
For each option, the quantile of the normal distribution is required. This can be done with a device-function. 
```
__device__ float cdf_normal(const float x){
    ...
}
```
Then, run 
```
BlackScholesKernel<<<500,1000>>>
```
## 2.2 Monte Carlo on the GPU
The `cuRAND` library generates random numbers for CUDA applications. Its documentation is visible [here](https://docs.nvidia.com/cuda/curand/index.html).
Example:  

```
curandSetPseudpRandomGenerator(seed, ...)
...
curandGenerateUniform(gen, devData, N);
...
cudaMemcpy(hostData, devData, numBytes, cudaMem)
```
Whenever the cuda program is using `cuRand`, it must be linked via `-l`, i.e.
```
nvcc -lcurand curand_test.cu -o curand_test
```

Look into reduction on the GPU on [here](https://www.youtube.com/watch?v=bpbit8SPMxU).

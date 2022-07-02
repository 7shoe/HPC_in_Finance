# Assignment 2: Matrix Multiplication

## 2.1 Single-Thread Matrix Multiplication
Multiply two matrices `A` and `B` each of size `NxN` with `N=500` into a new square matrx `C` os similar size.
- 2d vector `std::vector<std::vector<float>> A`: `2700`ms
- 2d double array `double A[N][N]`: `800`ms 
  (switching from vectors to static arrays cuts computational walltime in less than 3!)
- 2d float array `double A[N][N]`: `580`ms 
  (although double & floats are computed via 80bit registers on x86, memory bandwith seems to be a bottleneck speeding up float computation by ~20%) 

## 2.2 Multi-Thread Matrix Multiplication (on local machine)
#### 2.2.1 Install `OpenMP` on Mac
Download and run current OpenMP support for Mac OS from [Source](https://mac.r-project.org/openmp/).
```
curl -O https://mac.r-project.org/openmp/openmp-13.0.0-darwin21-Release.tar.gz
sudo tar fvxz openmp-13.0.0-darwin21-Release.tar.gz -C /
```
Move the downloaded files into the directory `/usr/local/include/openMP` as they might be loosely downloaded into `/usr/local/include`, e.g. the file path `usr/local/include/omp.h`. This messes up the better organized libraries that are neatly stored in single directories `/usr/local/include/eigen` or `/usr/local/include/boost`. Copy all loose files into `/usr/local/include/openMP`. 

Enable it in VSCode via `Cmd`+`Shift`+`P` and following the steps in [source](https://610yilingliu.github.io/2020/07/01/DebugCwithOpenmpinVscode/). In particular, 
add `"-fopenmp"` to `"args"` in `tasks` in the config JSON.

Then, including open `#include <omp.h>` will do the trick after adding `/usr/local/include/openMP` to `Include path` in the `C/C++ Configurations (UI)`. 

Run `openmp_helloworld.cpp` to confirm if it was successful. On the local machine, MacBook Pro (15-inch, 2017), `Hello World` will be printed 8 times.

#### 2.2.2 Actual parallelized matrix multiplication

(!) Just add `#pragma omp parallel for` ontop of the **outermost** loop, i.e.
```
#pragma omp parallel forfor (int i = 0; i < rows; ++i){ 
    for (int j = 0; j < columns; ++j){
        ...
    }
}
```
Note, that (accidentially) adding the `pragma` directive into the innermost loop increases runtime to a hefty `8000`ms! 

![Performance Table](/Assignment_B/performance.png)

Runtime: `152`ms! (An improved of a factor of 20x as compared to the naïve, single-threaded implementation).

## Homework: Explore Paralleliation Patterns
![Homework](/Assignment_B/homework.png)
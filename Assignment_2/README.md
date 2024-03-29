# Week 4: 

# 1. Random Number Generation in C++ with Intel's `MKL` Library
## Single-Stream Random Number Generator
- Intel Math Kernel Library (`MKL`) allows parallel mode for random number generator.  
- lot's of features in available when including `mkl_vsl.h`
- [extensive documentation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/statistical-functions/random-number-generators/vs-rng-usage-modelintel-onemkl-rng-usage-model.html)
- example uniform
```
VSLStreamStatePtr stream;

int seed =777;
vslNewStream(&stream, VSL_BRNG_MT19937, seed);
```

- example Gaussian
```
float mean = 0;
float stdev = 1.0f;

float* rands = new float[1000];

// example with stream argument for parallel implementation
vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream1, 10, rands2, mean, stdev);

vslDeleteStream(&stream1);
```
- `VSL_RNG_METHOD_GAUSSIAN_ICDF` [documentation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-vsnotes/top/testing-of-distribution-random-number-generators/continuous-distribution-random-number-generators/gaussian-vsl-rng-method-gaussian-icdf.html) is the random number generator of normal (Gaussian) distribution
- similarily `VSL_RNG_METHOD_GAUSSIAN_BOXMULLER` [documentation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-vsnotes/top/testing-of-distribution-random-number-generators/continuous-distribution-random-number-generators/gaussian-vsl-rng-method-gaussian-boxmuller.html) for 1 random number or `VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2` [documentation](https://www.google.com/search?q=VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2&rlz=1C5CHFA_enUS967DE970&oq=VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2&aqs=chrome..69i57.651j0j4&sourceid=chrome&ie=UTF-8) for 2 random numbers

- `BOXMULLER_1` and `_BOXMULLER_2` generate 1 and 2 random numbers, respectively
- delete the stream afterwards

## Parallel RNG
- use *block splitting* or *leap frogging*
- **block splitting** 
    - 1st stream generates:     $x_{1}$, $x_{k+1}$, $x_{2k+1}$ 
    - 2nd stream ...      :     $x_{2}$, $x_{k+2}$, $x_{2k+2}$ 
    - 3rd stream ...      :     $x_{3}$, $x_{k+3}$, $x_{2k+3}$ 
- **leap frogging** (aKa skip-ahead method)

- Note: leap frogging is not supported by all `VSL BRNG`

## Example: Simulation of $\pi$ via parallel random number generation
```
...
```

# 2. Python 
`cProfile`
`N` body problems and `Julia Set`
`High P

## Login
On Midway, login and request a compute node (e.g. for 1 hour) going forward
`sinteractive --account=finm32950 --time=1:0:0`
will forward us to `siebenschuh@midway3-0007 {Dictionary}` or something
Then run
```
module avail python
module load python/anaconda-2021.05

ipython
```
which opens a promt in which code can be run cell by cell.

### `cProfiler`
cProfile provides deterministic profiling of Python programs. Each profile is a set of statistics that describes how often and for how long various parts of the program executed. The Python standard library provides two different implementations. Namely, `cProfiler` and `profile`. We care for the former. 

`cProfile` is recommended for most users; it’s a C extension with reasonable overhead that makes it suitable for profiling long-running programs. Based 
[documentation](https://docs.python.org/3/library/profile.html)

Profile via 
```
python -m cProfile -s cumulative sample_file.py
```
only provides `ncalls` (number of function calls), `tottime` (total time spend) but it does not provide a line-by-line profiling.
On Midway, pip can be used to install libraries but they need to be installed into a local directory
```
cd .local/bin
pip install line_profiles
```

### `line_profiler`
Requires annotating the function we want to profile with annotator `@profile` 

## Python: A Dynamically Typed Language
C++ is static. At compile time a declared variable `float x = 1.0f;` is fixed as such. In Python, however, a variable `x` can be a float first `x = 1.0` and a `str` latter via `x = "Hello"`.

In Python, this dynamic functionality is realized with objects. These objects carry support operations. This results in additional overhead that is not useful when datatype discipline is applied.


## Cython
Allows usage of static C-type extensions for Python. The project's [website]()
Compiler is used to generate efficient C Code from Cython code which can be used in regular Python programs.

Imports
```
from setuptools import Extension,
from Cython.Build import cythonize
```

Example
```
cdef int a = 1
```
All C types are supported `bint` (instead of `bool`), `int, float`. Additionally, arrays and pointers are available.
```
cdef int k[10]
cdef double complex z, c
cdef unsigned int d
```
C functions and Python functions can call each other. *Fibbonaci* is a good example.
In the implemented example, `TreePricer_Numpy.py` which purely runs on a vectorized NumPy array. `TreePricer_Cython.pyx`, on the other hand, runs the same code in Cython. After profiling the entire function to be in C, it is imported in the main of `TreePricer_Numpy.py`. Both can be run to price `100` options. It takes `9.06 s` for the NumPy array and `8.94 s` for Cython. In contrast, only vectorizing the forward propagation of stocks takex `17x` longer.

To run the code:
```
python TreePricer_Numpy.py build_ext --inplace
```
The `--inplace` option creates the shared object file (with .so suffix) in the current directory.

## GIL: Global Interpreter Lock
`OpenMP` and `Cython` can be used together! However, only Cython types and no egular Python object can be run within any loop. 

```
from cython.parallel import prange
import numpy as np

with nogil:
    for i in prange(length_zs, schedule="static"):
        z = zs[i]
        ...

```

Then run `python setup.py`

## Example: `N'-Body Problem
The 2nd typical example for CPU-bound problems

See [source](https://www.oreilly.com/library/view/cython/9781491901731/ch04.html)


## Assignment 2 - README
@Author Carlo Siebenschuh
        University of Chicago
        siebenschuh@uchicago.edu
        July 11th, 2022

## RUN CODE
First, activate the correct module. Subsequently, compile (note that there are no dependencies, e.g. header files).
Moreover, no arguments are required as the program runs as specified.

```
  module load intel/2022.0
  icc -qopenmp assignment2.cpp -o assignment2.out
  ./assignment2.out
```

## IMPLEMENTATIONS   [Avg. Runtime +- stdev]
`assignment2_taskParallel.cpp`   [7.4ms  +- 1.9ms] --> parallelization (reduction) within method
`assignment2.cpp`                [5.6ms  +- 3.2ms] --> parallelization on task-level (each of the examples on on thread) (Submission)
`noParaAssignment2.cpp`          [0.0 ms +- 1.0ms] --> no parallelization

## CODE EXPLANATION
TL;DR: Leverage analytic representation of state-specific terminal stock price, unroll the backward propagation of the state-specific call price, exploit recombining property of stock price tree and the monotonocity of $(S * u^{k} * d^{N-k}-K)$ in the number of u to break loop-update of call price C0.

Details:
The program implements a tree-based pricing scheme for a European call option. The forward propagation (of the stock price dynamics) was replaced by the analytical form of the terminal stock price in the k-th state for N steps; namely, `S*std::pow(u, k)*(d, N-k)`. Furthermore, the backwad propagation is replaced since with $p_u = p_d = 0.5$ equal probability is assigned to each of the possible 2^N tracteories. This enables to update the call price C0 by adding each state-dependent option payout $(S - K)$ and updating `S *= d/u` when starting in state `k=N` (all up-movements) to (one down, N-1 up)- movement etc. Rather than summing over all 2^N possible trajectories (exponential in N),
the recombining property of the tree with its (N+1) terminal states is leverage and the number of combinations are counted for each terminal state via binomial coefficient (n over k). For numerical stability, in light of very large N, the logarithm of the binCoeff is updated additively (rather than the binomial coefficient itself multiplicatively). This, in turn, is faster than any numerical implementation of the binomial coefficient. Finally, since the states are looped from all-up towards all-down, the loop breaks as soon as `S<K`, as the remaining payouts will be `0.0`.
# Helpful Notes on the Task

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

```
## Assignment 1: Improved Black Scholes
European Call and Put options using Black Scholes formula. Compute the greeks: 
- delta
- gamma
- vega
- theta
- rho

![Greek Table](/Assignment_1/greek_table.png)

[Source](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#The_Options_Greeks)

### Login into *Midway* Server
- `ssh -Y siebenschuh@midway3.rcc.uchicago.edu`
- pw: as on okta for CNetID (requires push)
Run
- `module avail`
- `module list`
or, directly, 
- `module load intel/2022.0`

Use Use Intel complier (icc) to build the program
- `icc -O2 assignment1.cpp`

Then run the file 
- `./assignment1`

### Run the Assignment
Go to 
```
cd /home/siebenschuh
tar -xvf assignment1_siebenschuh_compr
ls Assignment1_siebenschuh

./assignment1
```

### Exact steps for Assignment 1
1. Generate random input values
2. Measure current time, <font color='red'>t1</font>
3. Price 1 million call, put options and greeks. Do NOT write results to console. Writing the results to console will slow down your program.  This step will be used to evaluate how fast your program ran.
4. Measure current time, <font color='red'>t2</font>
5. Write time taken (t2-t1) to console.
6. Using the functions used in step 3, price call, put and greeks for the following 3 options. Write results to console. Results will be used to verify correctness of your program

    (a)   S = 100; K = 100; r = 0.03; v = 0.3, T = 1
    
    (b)   S = 110; K= 100; r = 0.03, v= 0.3; T = 1
    
    (c)   S = 90; K = 100; r = 0.03; v = 0.3; T = 1
    
### A List of Improvements (in order of relative effectiveness)

#### Improvements 
- **Reducing the number of function calls from standard library**: exponential `std::exp()`, logarithmic `std::log()`, Gaussian error function `std::erf()`, and even power function `std::exp()` turned out to be major consumers of walltime. Miniming the call count drastically improved runtime. In particular, constants such as `std::sqrt(2.0)` slow the code down unnecessarily and are easily replaced. However, replacing functions by hand was not trivial (an online C implementation of the Gaussian error function was slower than `std::erf` even when `inline`d). Going forward, it is worth exploring how equivalent function definitions in `Boost` perform. ~1/3 of the runtime savings. 
- **put-call parity**: speeds computation up as it requires less evaluations of expensive functions (e.g. `std::exp()` and `std::log()`. In turn, it is a consequence of the former bullet point. Regardless, it shows that domain knowledge (economics) and mathematical rigor (BS prices must be arbitrage-free; in turn, put-call parity must hold) is invaluable when designing fast and correct software
- **global variables**: defining global variables and manipulating them through functions (without handing them over as arguments, neither value nor reference) turned out to be a fruitful improvement. In combination with replacing `std::` functions much runtime was saved.
- switching from `std::vector<double> S_vec(N)` to C arrays `double S_vec[N]` (that were globally defined) sped things up majorly.  
- `inline`ing function didn't seem to change runtime at all. In fact, it is unclear if the compiler even considers the keyword when optimizing.
- changing `if` conditions from `>` to `!=` (when possible) and switching the order of conditions helped a little. 

#### Neutral
- `double` to `float`: virtually no change although `float` is `32 bit` (IEEE 754 single precision) while `64 bit` IEEE 754 double precision. Likely explanation provided on [stackoverflow](https://stackoverflow.com/questions/4584637/double-or-float-which-is-faster#:~:text=Floats%20are%20faster%20than%20doubles,half%20the%20space%20per%20number.) and [quora](https://www.quora.com/Is-double-faster-than-float-in-C)
    - x86 processors have 80bit registers (neither float nor double). In turn, float and double are both extended (for free) to an internal 80-bit format, so both have the same performance (except for cache footprint / memory bandwidth
    - memory bandwith can be a bottleneck but doesn't have to be
    
#### Adverse
- Replacing the Gaussian quantile function was non-trivial. An idea was to piecewise approximate the quantile function and iterate the cases via `switch-cases`. However, this requires equality checks (not inequality). In turn, switching to `if else`-blocks was required. Unfortunately, this is very slow in the growing number of cases. Since the put/call prices and the Greeks are highly impacted by the accuracy of the quantules (linearly), I reverted back to `std::erf` and just tried to reduce the number of computations by introducing global variables such as `N_d1`, `N_d2` etc.
Another idea was to store these values in two sorted arrays (one for the quantile, another for the CDF function values). Looking up the CDF value is equivalent to looking up the largest value smaller than the input `x`. However, this binary search of a (sorted) array/`std::vector` didn't seem to be trivial and time didn't permit to do it. 

## Summary
The code compiled and ran beautifullty on `Midway`.
It ran in `61`ms (submitted version).
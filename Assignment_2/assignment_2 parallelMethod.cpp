#include <iostream>
#include <stdio.h>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <cmath>        /* exp */
#include <algorithm>
#include <omp.h>        /* open MP */

#define CORES 4

/**
 * @brief Logarithm of the factorial
 * 
 * @param k
 * @return double 
 */
double inline logFac(const int& k){
    double out = 0.0;
    for(int i=2; i < k+1; ++i){
        out += std::log(i);
    }
    return out;
}

/**
 * Price of a European Call option  (if the respective put price is available through the put-call parity)
 *
 * @param S current price of the underlying stock
 * @param K strike of the option as specified in the contract 
 * @param r rate of interest 
 * @param sigma volatility (aka standard deviation of the underlying stock), unobservable i.e. requires estimation
 * @param t remaining time until maturity of the option
 * @param N granularity of time-discretiziation 
 * @return CRR tree-infered price of the European Call option (as seen from long-position)
 */
double treeCall(double S, double K, double r, double sigma, double t, int N){ 

    // derived model parameters
    double delta_t = t / (1.0*N);
    double sqrt_delta_t = std::sqrt(delta_t);
    double u = std::exp((r - 0.5*sigma*sigma)*delta_t + sigma * sqrt_delta_t);
    double d = std::exp((r - 0.5*sigma*sigma)*delta_t - sigma * sqrt_delta_t);
    
    // vars for speed-up
    double prod = d / u;
    
    
    // forward propagation: stock price dynamics simulation 
    double C0 = 0.0;
    double PT = 0.0;

    // #pragma omp parallel for num_threads(CORES)
    S *= std::pow(u, N);                        // change --> ...(u, k)
    double logBinCoef = -N * std::log(2.0);      // change --> ...log(N over k)

    for(unsigned int i=N; i >= 0; --i){
        PT = (S * std::pow(u, i) * std::pow(d, N-i) - K);
        if(PT < 0){
            C0 += 0.0;
            break;
        }else{
            C0 += PT * std::exp(-N * std::log(2.0) + logFac(N) - logFac(N-i) - logFac(i));
        }
    }
    return C0 * std::exp(-r*t); 
}

int main() {

    /*
    S = 100;  K = 100;  r = 0.03;   v = 0.3,   T = 1;    N = 1000
    S = 110;  K= 100;   r = 0.03,   v = 0.3;   T = 1;    N = 1500
    S = 90;   K = 100;  r = 0j.03;  v = 0.3;   T = 1;    N = 100
    */

     // start the clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    double S_[3] = {100.0, 110.0, 90.0};
    double C_[3] = {0.0, 0.0, 0.0};
    int    N_[3] = {10, 15, 10};


    // do work() 
    for(unsigned int i=0; i < 3; i++){
        C_[i] = treeCall(S_[i], 100.0, 0.03, 0.3, 1.0,  N_[i]);
    }
    
    // stop time and show result
    // (1.) Measure time to price the following European call options
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n" << std::endl;

    // (2.) Write the option (call) prices to console: 
    for(unsigned int i=0; i < 3; i++){
        std::cout << "Tree price " << i << " : " << C_[i] << std::endl;
    }
}
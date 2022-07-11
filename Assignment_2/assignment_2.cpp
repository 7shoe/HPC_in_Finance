#include <iostream>
#include <vector>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <cmath>        /* exp */
#include <algorithm>

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
    double C0 = 0.0;
    double delta_t = t / (1.0*N);
    double sqrt_delta_t = std::sqrt(delta_t);
    double u = std::exp((r - 0.5*sigma*sigma)*delta_t + sigma * sqrt_delta_t);
    double d = std::exp((r - 0.5*sigma*sigma)*delta_t - sigma * sqrt_delta_t);
    
    // vars for speed-up
    double prod = d / u;
    double logBinCoef = std::log(0.5) * N;
    

    // update 
    S *= std::pow(u, N);

    // forward propagation: stock price dynamics simulation 
    for(unsigned int i=N; i >= 0; --i){
        if(S - K < 0){
            return C0 * std::exp(-r*t);
        }else{
            C0 += (S-K) * std::exp(logBinCoef);
            //binCoef *= i / (N_double - i + 1);
            logBinCoef += std::log(i) - std::log(1.0 + N - i);
            S *= prod;

            // debug
            //std::cout << "binCoef: " << (int)binCoef << std::endl;
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

    double C[3] = {0.0, 0.0, 0.0};
    double S[3] = {100.0, 110.0, 90.0};
    double N[3] = {1000, 1500, 100};

     // start the clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    // do work() 
    #pragma omp parallel for
    for (unsigned int i=0; i < 3; ++i){
        C[i] = treeCall(S[i], 100.0, 0.03, 0.3, 1.0,  N[i]);
    }
    
    // stop time and show result
    // (1.) Measure time to price the following European call options
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    // (2.) Write the option (call) prices to console: 
    for (unsigned int i=0; i < 3; ++i){
        std::cout << "\nTree price " << i << " : " <<  C[i] << std::endl;
    }
}
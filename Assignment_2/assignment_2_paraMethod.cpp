#include <iostream>
#include <stdio.h>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <cmath>        /* exp */
#include <algorithm>
#include <omp.h>        /* open MP */

/**
 * @brief Logarithm of the multiplicative factor multiplied to each conditional payout. It included
 *       - log. of binomial coefficient : log(n over k)
 *       - path probability : N*log(0.5)
 *       - time discount : -r*t
 * 
 * @param N 
 * @param k 
 * @param p 
 * @return double 
 */
double inline logProdCoef(const int& N, const int& k, const double& p, const double& r, const double& t){
    double out =  - (double)N * std::log(2) - r*t; 
    for(unsigned int i=1; i < k+1; ++i){
        out += std::log(1.0 + N - i) - std::log(i);
    }
    return out;
}

/**
 * @brief Computes the terminal value of the stock price
 * 
 * @param S 
 * @param u 
 * @param d 
 * @param N 
 * @param k 
 * @return double 
 */
double inline stateStockPrice(const double& S, const double& u, const double& d, const int& N, const int& k){
    double S_k = S * std::pow(u, k) * std::pow(d, N-k);
    return  S_k;
}
/**
 * @brief Tree-based pricing of call option via recombining tree structure
 * 
 * @param S 
 * @param K 
 * @param r 
 * @param t 
 * @param v 
 * @param N 
 * @return double 
 */
double treePricer(const double& S, const double& K, const double& r, const double& t, const double& v, const int& N){
    double delta_t = t / (1.0*N);
    double sqrt_delta_t = std::sqrt(delta_t);
    double u = std::exp((r - 0.5*v*v)*delta_t + v * sqrt_delta_t);
    double d = std::exp((r - 0.5*v*v)*delta_t - v * sqrt_delta_t);

    // compute 
    double CT = 0.0;
    #pragma omp parallel for shared(K, S, u, d, N, r, t) reduction(+:CT)
    for(int i=N; i >= 0; --i){
        double S_k = stateStockPrice(S, u, d, N, i);
        if(S_k > K){
            CT += (S_k - K) * std::exp(logProdCoef(N, i, 0.5, r, t)); 
        }
    }
    return CT; 
}

int main() {

    // start the clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    double out1 = treePricer(100.0, 100.0, 0.03, 1.0, 0.3, 1000);
    double out2 = treePricer(110.0, 100.0, 0.03, 1.0, 0.3, 1500);
    double out3 = treePricer(90.0,  100.0, 0.03, 1.0, 0.3,  100);
    
    // stop time and show result
    // (1.) Measure time to price the following European call options
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(1.) Elapsed time:\n" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n" << std::endl;

    std::cout << "(2.) Results: " << std::endl;
    std::cout << "Call 1: " << out1 << std::endl;
    std::cout << "Call 2: " << out2 << std::endl;
    std::cout << "Call 3: " << out3 << std::endl;
}
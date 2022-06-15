#include <stdio.h>      /* printf */
#include <math.h>       /* std::erf(double): Gaussian error function */
#include <cmath>        /* std::log(double) */
#include <iostream>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <vector>

using namespace std;

/**
 * Cummumlative distribution function (CDF) of the univariate normal (aka Gaussian) distribution
 *
 * @param x real-valued argument 
 * @return probability P(X <= x) given X ~ Normal(0,1)
 */
float N_CDF(float x){
    return 0.5 * (1.0 + std::erf(x / std::sqrtf(2.0)));
}


/**
 * Draw uniform number from interval ranging from low to hi-value. 
 *
 * @param low minimum value 
 * @param hi maximum value 
 * @return Samle random number uniformly from interval [low, hi]
 */
float random_data(float low, float hi)  {

   float r = (float)rand() / (float)RAND_MAX;

   return low + r * (hi - low);

}

/**
 * Price of a European Call option  (if the respective put price is available through the put-call parity)
 *
 * @param S current price of the underlying stock
 * @param K strike of the option as specified in the contract 
 * @param r rate of interest 
 * @param sigma volatility (aka standard deviation of the underlying stock), unobservable i.e. requires estimation
 * @param t remaining time until maturity of the option
 * @return model-infered price of the European Call option (as seen from long-position)
 */
float call(float S, float K, float r, float t, float sigma, float Put_p=0.0) {
    if(Put_p > 0){
        return S + Put_p - std::expf(-r*t) * K; 
    }else{
        float d_1 = (std::logf(S/K) + r + t * std::powf(sigma, 2) / 2.0) / (sigma * std::sqrtf(t));
        return S * N_CDF(d_1) - std::expf(-r*t) * K * N_CDF(d_1 - sigma * std::sqrtf(t));
    }
}

/**
 * Price of a European Put option (if the respective call price is available through the put-call parity)
 *
 * @param S current price of the underlying stock
 * @param K strike of the option as specified in the contract 
 * @param r rate of interest (idealized)
 * @param sigma volatility (aka standard deviation of the underlying stock), unobservable i.e. requires estimation
 * @param t remaining time until maturity of the option
 * @return model-infered price of the European Put option (as seen from long-position)
 */
float put(float S, float K, float r, float t, float sigma, float Call_p=0.0) {
    
    if(Call_p > 0.0){
        return K * std::expf(-r * t) - S + Call_p;
    }else{
        float d_1 = (std::logf(S/K) + r + t * std::powf(sigma, 2) / 2.0) / (sigma * std::sqrtf(t));
        return std::expf(-r*t) * K * N_CDF(sigma * std::sqrtf(t) - d_1) - S * N_CDF(-d_1);
    }
}

int main ()
{
    // number of options to be priced
    int N = (int)1e6;                 // 10E+6

    // preparation: generate random inputs:
    vector<float> S_vec, K_vec, r_vec, t_vec, sigma_vec;

    // reserve space 
    S_vec.reserve(N);
    K_vec.reserve(N);
    r_vec.reserve(N);
    t_vec.reserve(N);
    sigma_vec.reserve(N);

    // fill vectors
    for (int i = 0; i < N; ++i)
        S_vec.push_back(random_data(0.1,   1000.0));
        K_vec.push_back(random_data(10.0,  250.0));
        r_vec.push_back(random_data(0.0,   0.2));
        t_vec.push_back(random_data(0.05,  5.0));
        sigma_vec.push_back(random_data(0.05,  0.8));

    // output target
    float pCall, pPut;

    // stop the time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //do-work();
    std::cout << put(60.0, 40.0, 0.05, 2.0, 0.2) << std::endl;
    for (int i = 0; i < N; i++) {
        pCall = call(S_vec[i], K_vec[i], r_vec[i], t_vec[i], sigma_vec[i]); 
        pPut  = put(S_vec[i], K_vec[i], r_vec[i], t_vec[i], sigma_vec[i], pCall); 
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // std::cout << put(60.0, 40.0, 0.05, 2.0, 0.2) << std::endl;

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms";
}

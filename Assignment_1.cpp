#include <stdio.h>      /* printf */
//#include <math.h>       /* std::erf(double): Gaussian error function */
#include <iostream>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <vector>
#include <cmath>        /* std::log(double) */

#define N 1000000

// global variable
double exp_min_t_r, sigma_sqrt_t, d_1, N_d1, N_d2, Np_d1, N_min_d2, phi_d1, sqrt_t, theta_summand; 

double S_vec[N]; 
double K_vec[N];  
double r_vec[N];  
double t_vec[N];  
double sigma_vec[N];  

double Greek_arr[8];

/**
 * Probability density function (PDF) of the univariate normal (aka Gaussian) distribution
 *
 * @param x real-valued argument 
 * @return density f(x) given X ~ Normal(0,1)
 */
double inline phi(const double& x){
    return 0.39894228040 * std::exp(-0.5*x*x);
}

/**
 * Cummumlative distribution function (CDF) of the univariate normal (aka Gaussian) distribution
 *
 * @param x real-valued argument 
 * @return probability P(X <= x) given X ~ Normal(0,1)
 */
double inline N_CDF(const double& x){
    return 0.5 + 0.5*std::erff(x * 0.7071067812);
}

/**
 * Draw uniform number from interval ranging from low to hi-value. 
 *
 * @param low minimum value 
 * @param hi maximum value 
 * @return Samle random number uniformly from interval [low, hi]
 */
double random_data(double low, double hi)  {

   double r = (double)rand() / (double)RAND_MAX;

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
double inline call(const double& S, const double& K, const double& r, const double& t, const double& sigma, const double& Put_p=0.0) {
    if(Put_p != 0){
        return S + Put_p - exp_min_t_r * K; 
    }else{
        exp_min_t_r = std::exp(-r*t);
        sqrt_t = std::sqrt(t);
        sigma_sqrt_t = sigma * sqrt_t;
        d_1 = (std::log(S/K) + (r + 0.5 * sigma * sigma)*t) / sigma_sqrt_t;
        phi_d1 = phi(d_1);
        N_d1 = N_CDF(d_1);
        N_d2 =  N_CDF(d_1 - sigma_sqrt_t);
        return S * N_d1 - exp_min_t_r * K * N_d2;
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
double inline put(const double& S, const double& K, const double& r, const double& t, const double& sigma, const double& Call_p=0.0) {
    N_min_d2 = N_CDF(sigma_sqrt_t - d_1);
    if(Call_p == 0.0){
        exp_min_t_r = std::exp(-r*t);
        sqrt_t = std::sqrt(t);
        sigma_sqrt_t = sigma * sqrt_t;
        d_1 = (std::log(S/K) + (r + 0.5 * sigma * sigma)*t) / sigma_sqrt_t;
        return exp_min_t_r * K * N_min_d2 - S * N_CDF(-d_1);
    }else{
        return K * exp_min_t_r - S + Call_p;
    }
}

/**
 * Greeks (Delta (C/P), Gamma, Vega, Rho(C/P), Theta(C/P))
 *
 * @param S current price of the underlying stock
 * @param K strike of the option as specified in the contract 
 * @param r rate of interest 
 * @param sigma volatility (aka standard deviation of the underlying stock), unobservable i.e. requires estimation
 * @param t remaining time until maturity of the option
 * @return array of model-infered Greeks
 */
void inline greeks(const double& S, const double& K, const double& r, const double& t, const double& sigma) {
    // Delta
    Greek_arr[0] = N_d1;          // Call 
    Greek_arr[1] = N_d1 - 1.0;    // Put
    // Gamma
    Greek_arr[2] = phi_d1 / (S * sigma_sqrt_t); 
    // Vega
    Greek_arr[3] = S*phi_d1*sqrt_t;
    // Rho
    Greek_arr[4] =  K * t * exp_min_t_r * N_d2;              // Call
    Greek_arr[5] =  -1.0 * K * t * exp_min_t_r * N_min_d2;   // Put
    // Theta
    theta_summand = -0.5 * S * phi_d1 * sigma / sqrt_t;
    Greek_arr[6] = theta_summand -r * K * exp_min_t_r * N_d2;        // Call
    Greek_arr[7] = theta_summand + r * K * exp_min_t_r * N_min_d2;   //Put
}

int main ()
{
    // fill vectors
    for (int i = 0; i < N; ++i){
        S_vec[i] = random_data(0.1,   1000.0);
        K_vec[i] = random_data(10.0,  250.0);
        r_vec[i] = random_data(0.0,   0.2);
        t_vec[i] = random_data(0.05,  5.0);
        sigma_vec[i] = random_data(0.05,  0.8);
    }

    // sstd::cout << "S_vec.size(): " << S_vec.size() << std::endl;

    // output target
    double pCall, pPut;

    // stop the time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //do-work();
    //std::cout << put(60.0, 40.0, 0.05, 2.0, 0.2) << std::endl;
    for (int i = 0; i < N; i++) {
        pCall = call(S_vec[i], K_vec[i], r_vec[i], t_vec[i], sigma_vec[i]); 
        pPut  = put(S_vec[i], K_vec[i], r_vec[i], t_vec[i], sigma_vec[i], pCall); 
        greeks(S_vec[i], K_vec[i], r_vec[i], t_vec[i], sigma_vec[i]);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // Part A   
    std::cout << "- - - - - - - - - - - - - - " << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    std::cout << "- - - - - - - - - - - - - - " << std::endl;

    // Part B
    std::cout << "\n\nExamples: " << std::endl;
    std::cout << call(100.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    std::cout << put(100.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    greeks(100.0, 100.0, 0.03, 1.0, 0.3); 
    for (int j=0; j < 7; j++){
        std::cout << Greek_arr[j] << ", "; 
    }
    std::cout << Greek_arr[7] << "\n" << std::endl; 


    std::cout << call(110.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    std::cout << put(110.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    greeks(110.0, 100.0, 0.03, 1.0, 0.3); 
        for (int j=0; j < 7; j++){
        std::cout << Greek_arr[j] << ", "; 
    }
    std::cout << Greek_arr[7] << "\n" << std::endl; 

    std::cout << call(90.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    std::cout << put(90.0, 100.0, 0.03, 1.0, 0.3) << std::endl; 
    greeks(90.0, 100.0, 0.03, 1.0, 0.3); 
        for (int j=0; j < 7; j++){
        std::cout << Greek_arr[j] << ", "; 
    }
    std::cout << Greek_arr[7] << std::endl;  

}
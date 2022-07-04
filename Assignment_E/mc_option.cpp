#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>       /* measure time */
#include <cmath>        /* std::log(double) */
#include <random>       /* standard*/
#include <map>

#define N   1000000

double C[N];
double S_arr[N];
double K_arr[N];
double t_arr[N];
double r_arr[N];
double sigma_arr[N];

// log-Normal distribution
std::random_device rd;
std::mt19937 gen(rd());

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
 * Monte Carlo Simulation of European Option Price by sampling lognormal terminal stock price
 *
 * @param S initial stock price 
 * @param K strike price
 * @param r risk-free interest rate
 * @param t (remaining) time to maturity
 * @param sigma volatility/standard deviation
 * @param M number of lognormal stock prices drawn for MC simulation
 * @return Samle random number uniformly from interval [low, hi]
 */
double MC_call(const double& S, const double& K, const double& t, const double& r, const double& sigma, const int& M){
    // set LN parameters
    double m = std::log(S) + (r - 0.5 * sigma * sigma) * t;
    double v = sigma * std::sqrt(t);
    std::lognormal_distribution<> d(m, v);

    // Monte Carlo Simulation of Option Price
    double C_hat = 0.0;
    for(int n=0; n<M; ++n) {
        C_hat += std::max(0.0, d(gen) - K);
    }
    C_hat *= std::exp(-r*t) / M;

    return C_hat;
}


int main(){

    // random inputs
    for(int k=0; k < N; ++k){
        S_arr[k] = random_data(0.1,   1000.0);
        K_arr[k] = random_data(10.0,  250.0);
        r_arr[k] = random_data(0.0,   0.2);
        t_arr[k] = random_data(0.05,  5.0);
        sigma_arr[k] = random_data(0.05,  0.8);
    }
    
    // simulate 1mio option parameters
 
    // start 
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Loop: 1mio options
    for(int k=0; k < N; ++k){
        C[k] = MC_call(S_arr[k], K_arr[k], t_arr[k], r_arr[k], sigma_arr[k], 100);
    }

    // stop
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // Output
    std::cout << "C_hat: " << C[10] << std::endl;

    // Report 
    std::cout << "- - - - - - - - - - - - - - " << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    std::cout << "- - - - - - - - - - - - - - " << std::endl;
    
}
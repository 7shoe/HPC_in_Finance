#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>       /* measure time */
#include <cmath>        /* std::log(double) */
#include <random>       /* standar*/
#include <map>

#define N   10000000

int main(){

    // Input 
    double S = 50;
    double K = 45;
    double t = 1.0;
    double r = 0.05;
    double sigma = 0.3;

    // LN parameters
    double m = std::log(S) + (r - 0.5*sigma*sigma)*t;
    double v = sigma * std::sqrt(t);

    // specify log-Normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<> d(m, v);
 
    // start 
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Monte Carlo Simulation of Option Price
    double C_hat = 0.0;
    for(int n=0; n<N; ++n) {
        C_hat += std::max(0.0, d(gen) - K);
    }
    C_hat *= std::exp(-r*t) / N;

    // stop
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // Output
    std::cout << "C_hat: " << C_hat << std::endl;

    // Report 
    std::cout << "- - - - - - - - - - - - - - " << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    std::cout << "- - - - - - - - - - - - - - " << std::endl;
    
}
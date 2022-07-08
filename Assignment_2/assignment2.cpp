#include <stdio.h>      /* printf */
//#include <math.h>       /* std::erf(double): Gaussian error function */
#include <iostream>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <vector>
#include <cmath>        /* std::log(double) */
#include <omp.h>        /* open MP */

/**
 * Random number generator, X~Unif([low, hi])
 *
 * @low Lowest possible value
 * @hi Largest possible value
 */
float random_data(float low, float hi){
    float r = (float)rand() / (float)RAND_MAX;return low + r * (hi - low);
    return low + r * (hi - low);
}

double Binomial_BS(const double& S, const double& K, const double& r, const double& t, const double& sigma, const int& N){
    // initialize parameters
    double u = (r - 0.5 * sigma * sigma) * t + sigma * std::sqrt(t);
    double d = (r - 0.5 * sigma * sigma) * t - sigma * std::sqrt(t);

    // populate terminal noded of 
    std::vector<double> S_vec(N+1, S * std::pow(u, N));
    double u_pow_k;
    for(int k=1; k < N+1; ++k){
        S_vec[k] *= std::pow(d, 2.0*(N-k));
    }

    // 

    // DEBUG
    std::cout << "S_vec[0]: " << S_vec[0] << ", S_vec[N]: " << S_vec[N] << std::endl;

    // Backpropagate Tree
    return 0.0;

}


/*
Using OpenMP, write a multithreaded binomial pricer.

1) Measure time to price the following European call options and write it to console.

a)   S = 100; K = 100; r = 0.03;  v = 0.3,  T = 1;  N = 1000
b)   S = 110; K= 100;  r = 0.03,  v = 0.3;  T = 1;  N = 1500
c)   S = 90;  K = 100; r = 0j.03; v = 0.3;  T = 1;  N = 100

(N number of steps)

2. Write the option (call) prices to console.

Submission: submit a tar file (similar to what was done for Assignment 1)

*/


int main() {

    std::cout << "..." << std::endl;

    Binomial_BS(100, 100, 0.03, 1.0, 0.3, 1000);

    return 0;
}
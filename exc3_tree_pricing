#include <iostream>
#include <vector>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <cmath>        /* exp */
#include <algorithm>

using namespace std; 

#define N    20
#define RUNS 10E6

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

/**
 * self-made max-function that returns biggest number 
 *
 * @num1 1st input number 
 * @num2 1st input number 
 */
float maxF(float num1, float num2) {
    if (num1 >= num2) {
        return num1;
    } else {
        return num2;
    }
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
float treeCall(float S, float K, float r, float t, float sigma){ 

    // derived model parameters
    float delta_t = t / (1.0*N);
    double u = std::exp((r - 0.5*sigma*sigma)*delta_t + sigma*std::sqrt(delta_t));
    double d = std::exp((r - 0.5*sigma*sigma)*delta_t - sigma*std::sqrt(delta_t));
    
    // DP Table initialization
    vector<vector<float>> T(N+1, vector<float> (N+1, 0));
    T[0][0] = S;
    
    // forward propagation: stock price dynamics simulation 
    for (int i=1; i < T.size(); i++) {
        for (int j=0; j < i+1; j++) {
            T[i][j] = S*std::pow(u, j)*std::pow(d, i-j);
        }
    }
    
    // calculate (conditional) call option payout at terminal nodes
    for (int j=0; j < T.size(); j++) {
            T[T.size()-1][j] = maxF(T[T.size()-1][j] - K, 0.0);
    }
    
    // backward propagation: call price calculation
    for (int i=T.size()-2; i > -1; i--) {
        for (int j=0; j < i+1; j++) {
            T[i][j] = 0.5*(T[i+1][j] + T[i+1][j+1]) * std::exp(-r * delta_t);
        }
    }
    
    return T[0][0];
    
    
}

int main() {
    
     // stop the time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    // do shit() 
    for (int i=0; i < RUNS; i++) {
        
        treeCall(random_data(0.01, 1000.0), random_data(1.0, 100.0), random_data(0.01, 0.1), random_data(0.1, 10.0), random_data(0.01, 0.5));
    }
    
    
    // stop time and show result
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    
}
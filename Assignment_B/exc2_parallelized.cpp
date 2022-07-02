#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */
#include <stdlib.h>

using namespace std;

#define N 500

float A[N][N];
float B[N][N];
float C[N][N]; 


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
 * Naïve matrix multiplication of two input matrices of the same size with complexity O(N^3) 
 *
 * @A Square matrix (NxN) as a 2-d float vector
 * @B Square matrix (NxN) as a 2-d float vector
 */
void matrixMult() { 
    #pragma omp parallel for
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k=0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    
     // stop the time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    
    // print the output?
    bool print_flag = false; 
    
    // randomly assign values to matrices
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            A[i][j] = random_data(-10.0, 10.0);
            B[i][j] = random_data(-10.0, 10.0);
        }
    }
    
    // actual matrix multiplication
    matrixMult();
    
    // stop time and show result
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    
    // print if needed
    if(print_flag) {
        for (int i=0; i < N; i++) {
            for (int j=0; j < N; j++) {
                std::cout << C[i][j] << "  ";
                
            }
            std::cout << std::endl;
        }
    }
    
}
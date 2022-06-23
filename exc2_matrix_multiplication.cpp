#include <iostream>
#include <vector>
#include <chrono>       /* measure time */
#include <cstdlib>      /* rand() */

#define N 100

using namespace std; 


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
 * @A Square matrix (NxN) as a 2-d double vector
 * @B Square matrix (NxN) as a 2-d double vector
 */
vector<vector<float>> matrixMult(vector<vector<float>> A, vector<vector<float>> B) { 

    vector<vector<float>> C(N, vector<float> (N, 0));

    for (int i=0; i < A.size(); i++) {
        for (int j=0; j < A[0].size(); j++) {
            C[i][j] = 0.0;
            for (int k=0; k < A.size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    
     // stop the time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    
    // print the output?
    bool print_flag = false; 
    
    // declare matrices as 2d vector objects
    vector<vector<float>> A(N, vector<float> (N, 0));
    vector<vector<float>> B(N, vector<float> (N, 0));
    
    // randomly assign values to matrices
    for (int i=0; i < A.size(); i++) {
        for (int j=0; j < A[0].size(); j++) {
            A[i][j] = random_data(-10.0, 10.0);
            B[i][j] = random_data(-10.0, 10.0);
        }
    }
    
    // actual matrix multiplication
    vector<vector<float>> C = matrixMult(A, B);
    
    // stop time and show result
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    
    // print if needed
    if(print_flag) {
        for (int i=0; i < C.size(); i++) {
            for (int j=0; j < C[0].size(); j++) {
                std::cout << C[i][j] << "  ";
                
            }
            std::cout << std::endl;
        }
    }
    
}
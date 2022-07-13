#include <stdio.h>
#include <iostream>
#include <stdlib.h>

void add(float* a, float* b, float* c, int N){
        for (int i=0; i<N; ++i){
                c[i] = a[i] + b[i];
        }
}


int main(){

        int N = 1024;
        float* h_a, *h_b, *h_c;

        // allocate memory
        h_a = (float *)malloc(N*sizeof(float));
        h_b = (float *)malloc(N*sizeof(float));
        h_c = (float *)malloc(N*sizeof(float));

        // init data
        for (int i=0; i<N; ++i){
                h_a[i] = (float)N-i-1;
                h_b[i] = (float)i-N+i+2;
        }

        // use function
        add(h_a, h_b, h_c, N);

        // reduce to sum
        float s = 0.0;
        for (int i=0; i<N; ++i){
                s+=h_c[i];
        }

        std::cout << "Sum: " << s << std::endl;

        // free up memory
        free(h_a);
        free(h_b);
        free(h_c);
}
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include <chrono>       /* measure time */



int main(){




    // start time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // stop time 
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    // measure time
    std::cout << "cudaMemcpy: host->device: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n" << std::endl;


}
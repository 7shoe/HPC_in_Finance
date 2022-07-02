#include<stdio.h>
#include <omp.h>
#include <iostream>

int main() {
    #pragma omp parallel
    std::cout << "Hello, world" << std::endl;

}
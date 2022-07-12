

int main(){
        // declare host memory and put data in
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

        // declare device memory
        float* d_a, *d_b, *d_c;

        cudaMalloc((float**)&d_a, N*sizeof(float));
        cudaMalloc((float**)&d_b, N*sizeof(float));
        cudaMalloc((float**)&d_c, N*sizeof(float));

        // copy data to device
        cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

        // use device function
        gpuAdd<<<16, 64>>>(d_a, d_b, d_c, N);

        cudaThreadSynchronize();

        // copy result from device back to host
        cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

        // reduce to sum
        float s = 0.0;
        for (int i=0; i<N; ++i){
                s+=h_c[i];
        }

        // free up device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // free host memory
        free(h_a);
        free(h_b);
        free(h_c);


        std::cout << "Sum: " << s << std::endl;


        // free up memory
        free(h_a);
        free(h_b);
        free(h_c);
}
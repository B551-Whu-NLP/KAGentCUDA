#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int array_size = 10000000;
    float* A_host = (float*)malloc(array_size * sizeof(float));
    float* B_host = (float*)malloc(array_size * sizeof(float));
    float* C_host = (float*)malloc(array_size * sizeof(float));
    
    for (int i = 0; i < array_size; ++i) {
        A_host[i] = i + 1.0;
        B_host[i] = (i + 1.0) * 2.0;
        C_host[i] = 0.0;
    }
    
    clock_t start = clock();
    
    for (int i = 0; i < array_size; i++) {
        C_host[i] = A_host[i] / B_host[i];
    }
    
    clock_t end = clock();
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        FILE *fp = fopen("c_221_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", C_host[i]);
            }
            fclose(fp);
        }
    }
    
    free(A_host);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    free(B_host);
    free(C_host);
    
    return 0;
}
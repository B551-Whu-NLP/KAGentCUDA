#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int N = 1000;
    float clamp_min = -100.0f;
    float clamp_max = 100.0f;
    float* h_X = (float*)malloc(N * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    for (int i = 0; i < N; ++i) {
        h_X[i] = (float)(i - 500);
    }
    
    clock_t start = clock();
    
    for (int i = 0; i < N; ++i) {
        h_X[i] = fminf(clamp_max, fmaxf(clamp_min, h_X[i]));
    }
    
    clock_t end = clock();
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (verbose) {
        FILE *fp = fopen("c_251_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_X[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_X);
    printf("Time: %.2fms\n", time_taken * 1000);
    return 0;
}
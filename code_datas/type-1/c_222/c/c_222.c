#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void SAXPY(int len, float a, float* d_x, float* d_y) {
    for (int i = 0; i < len; i++) {
        d_y[i] = d_x[i] * a + d_y[i];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000;
    float a = 0.5;
    float* h_x = (float*)malloc(array_size * sizeof(float));
    float* h_y = (float*)malloc(array_size * sizeof(float));
    
    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i + 1.0;
        h_y[i] = (i + 1.0) * 2.0;
    }
    
    clock_t start = clock();
    
    SAXPY(array_size, a, h_x, h_y);
    
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_222_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_y[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_x);
    free(h_y);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}
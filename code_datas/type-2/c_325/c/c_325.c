#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void array_rowKernel(double *input, double *output, int length, int verbose, const char* file_name) {
    int xCuda, yCuda;
    clock_t start = clock();
    
    for (xCuda = 0; xCuda < length; xCuda++) {
        for (yCuda = 0; yCuda < length; yCuda++) {
            int idx = yCuda * length + xCuda;
            if (xCuda == 0 || xCuda == length - 1) {
                output[idx] = 0;
            } else {
                output[idx] = input[idx];
                output[idx] += xCuda == 0 ? 0 : input[idx - 1];
                output[idx] += xCuda == length - 1 ? 0 : input[idx + 1];
            }
        }
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
    
    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", file_name);
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < length; j++) {
                    fprintf(fp, "%.2f ", output[length * i + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
}

int main(int argc, char *argv[]) {
    int length = 1000;
    int verbose = 0;
    const char* file_name = "c_325";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }
    
    double *input = (double *)malloc(length * length * sizeof(double));
    double *output = (double *)malloc(length * length * sizeof(double));
    
    for (int i = 0; i < length * length; i++) {
        input[i] = (i + 1) % 100; 
    }
    
    array_rowKernel(input, output, length, verbose, file_name);
    
    free(input);
    free(output);
    
    return 0;
}
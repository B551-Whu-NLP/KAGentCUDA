#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void PSIfill(float* array, int conv_length, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = array[i % conv_length];
    }
}

int main(int argc, char *argv[]) {
    int arraySize = 1000000;
    int convLength = 500000;
    float* h_array = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = (float)(i);
    }

    clock_t start = clock();
    PSIfill(h_array, convLength, arraySize);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_array[i]);
            }
            fprintf(fp, "\nTime: %.2f(ms)\n", time_taken);
            fclose(fp);
        }
    }
    
    free(h_array);
    return 0;
}
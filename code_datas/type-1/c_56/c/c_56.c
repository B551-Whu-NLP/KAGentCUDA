#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void compute_array_square(float *array, float *outArray, int size) {
    clock_t start = clock();

    for (int i = 0; i < size; i++) {
        outArray[i] = array[i] * array[i];
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    const int size = 50000000; // 进一步增大数组大小
    float* h_array = (float*)malloc(size * sizeof(float));
    float* h_outArray = (float*)malloc(size * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    for (int i = 0; i < size; ++i) {
        h_array[i] = (float)(i);
    }

    compute_array_square(h_array, h_outArray, size);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_56_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "outArray[%d]: %f\n", i, h_outArray[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_array);
    free(h_outArray);
    return 0;
}
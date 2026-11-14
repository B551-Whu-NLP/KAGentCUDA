#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void test(float* input, const int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (tid == 0) {
            input[tid] = 0;
        }
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    float* h_input = (float*)malloc(arraySize * sizeof(float));
    
    srand(42);
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = rand() % 100;
    }

    clock_t start = clock();
    
    test(h_input, arraySize);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_10_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_input[i]);
            }
            fclose(file);
        }
    }

    free(h_input);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sum_array(float* a, float* b, float* c, const int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));
    
    srand(42);
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(rand() % 100);
        h_b[i] = (float)(rand() % 100);
    }

    clock_t start = clock();

    // 增加迭代次数
    for (int iter = 0; iter < 100; iter++) {
        sum_array(h_a, h_b, h_c, arraySize);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_24_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}
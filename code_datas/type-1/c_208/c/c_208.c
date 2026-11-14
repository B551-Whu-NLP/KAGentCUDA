#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void test1(float* input, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (input[tid * 4] != 0) {
            input[tid * 4] = 0;
        }
    }
}

int main(int argc, char** argv) {
    const int array_size = 1000000; // 增大数组大小
    float* input_host = (float*)malloc(array_size * 4 * sizeof(float));
   
    for (int i = 0; i < array_size * 4; ++i) {
        input_host[i] = i + 1.0;
    }

    clock_t start_time = clock();

    // 增加迭代次数
    for (int iter = 0; iter < 1000; iter++) {
        test1(input_host, array_size);
    }

    clock_t end_time = clock();

    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_208_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", input_host[i]);
            }
            fclose(file);
        }
    }

    free(input_host);
  

    return 0;
}
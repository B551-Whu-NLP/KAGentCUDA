#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void evenoddincrement(float* g_data, int even_inc, int odd_inc, int size) {
    for (int tx = 0; tx < size; tx++) {
        if ((tx % 2) == 0) {
            g_data[tx] += even_inc;
        }
        else {
            g_data[tx] += odd_inc;
        }
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000; // 增大数组大小
    int even_inc = 2;
    int odd_inc = 3;
    float* h_data = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_data[i] = (float)(i);
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 100; iter++) {
        evenoddincrement(h_data, even_inc, odd_inc, arraySize);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_48_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_data[i]);
            }
            fclose(fp);
        }
    }

    free(h_data);
    return 0;
}
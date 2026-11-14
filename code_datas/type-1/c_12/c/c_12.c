#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void dot(float* c, float* a, float* b, int size) {
    int t_id;
    for (t_id = 0; t_id < size; t_id++) {
        c[t_id] = a[t_id] * b[t_id];
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(i);
        h_b[i] = (float)(2 * i);
    }

    clock_t start = clock();

    dot(h_c, h_a, h_b, arraySize);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_12_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%f ", h_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
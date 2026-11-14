#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vadd(const float* a, const float* b, float* c, const unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    unsigned int vector_size = 10000000; // 增加向量大小
    float* h_a, * h_b, * h_c;
    h_a = (float*)malloc(vector_size * sizeof(float));
    h_b = (float*)malloc(vector_size * sizeof(float));
    h_c = (float*)malloc(vector_size * sizeof(float));

    for (unsigned int i = 0; i < vector_size; ++i) {
        h_a[i] = i * 1.5f;
        h_b[i] = i * 2.0f;
    }

    clock_t start = clock();

    vadd(h_a, h_b, h_c, vector_size);

    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_227_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    } 

    free(h_a);
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_b);
    free(h_c);

    return 0;
}
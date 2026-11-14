#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vectorAdd(double* a, double* b, double* c, int vector_size) {
    for (int idx = 0; idx < vector_size; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char* argv[]) {
    int vector_size = 100000000; // 增大向量大小
    double* h_a, * h_b, * h_c;
    h_a = (double*)malloc(vector_size * sizeof(double));
    h_b = (double*)malloc(vector_size * sizeof(double));
    h_c = (double*)malloc(vector_size * sizeof(double));

    for (int i = 0; i < vector_size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    clock_t start = clock();

    vectorAdd(h_a, h_b, h_c, vector_size);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_224_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    }


    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
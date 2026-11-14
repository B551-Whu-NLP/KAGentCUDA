#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void intSubtract(int* c, const int* a, const int* b, const unsigned int d) {
    for (unsigned int i = 0; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

int main(int argc, char* argv[]) {
    unsigned int vector_size = 10000000; // 增大向量大小
    int* h_a, * h_b, * h_c;
    h_a = (int*)malloc(vector_size * sizeof(int));
    h_b = (int*)malloc(vector_size * sizeof(int));
    h_c = (int*)malloc(vector_size * sizeof(int));

    for (unsigned int i = 0; i < vector_size; ++i) {
        h_a[i] = i * 2;
        h_b[i] = i;
    }

    clock_t start_time = clock();

    intSubtract(h_c, h_a, h_b, vector_size);

    clock_t end_time = clock();

    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_228_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);

    printf("Time: %.2fms\n", time_taken * 1000);

    return 0;
}
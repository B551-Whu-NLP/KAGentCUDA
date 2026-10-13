#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void intAdd(int* c, const int* a, const int* b, const unsigned int d) {
    for (unsigned int i = 0; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    unsigned int vector_size = 10000000;
    int* h_a, * h_b, * h_c;
    h_a = (int*)malloc(vector_size * sizeof(int));
    h_b = (int*)malloc(vector_size * sizeof(int));
    h_c = (int*)malloc(vector_size * sizeof(int));

    for (unsigned int i = 0; i < vector_size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    clock_t start = clock();

    intAdd(h_c, h_a, h_b, vector_size);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_225_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    
    printf("Time: %.2fms\n", time_taken * 1000);
    free(h_b);
    free(h_c);

    return 0;
}
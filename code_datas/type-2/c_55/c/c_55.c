#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void transposition(int *vector, int *transposed, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            transposed[i + j * size] = vector[j + i * size];
        }
    }
}

int main(int argc, char *argv[]) {
    const int size = 10;
    int* h_vector = (int*)malloc(size * size * sizeof(int));
    int* h_transposed = (int*)malloc(size * size * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size * size; ++i) {
        h_vector[i] = i;
    }

    clock_t start = clock();

    transposition(h_vector, h_transposed, size);

    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < size; ++j) {
                    fprintf(fp, "%d\t", h_vector[j + i * size]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < size; ++j) {
                    fprintf(fp, "%d\t", h_transposed[j + i * size]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_vector);
    free(h_transposed);
    return 0;
}
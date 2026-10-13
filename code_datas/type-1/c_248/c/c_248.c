#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void reverseArrayBlock(int* d_out, int* d_in, int size) {
    for (int i = 0; i < size; i++) {
        d_out[i] = d_in[size - 1 - i];
    }
}

int main(int argc, char *argv[]) {
    int size = 10000000; // 进一步增大数组大小
    int* h_in = (int*)malloc(size * sizeof(int));
    int* h_out = (int*)calloc(size, sizeof(int));

    srand(42);
    for (int i = 0; i < size; ++i) {
        h_in[i] = rand() % 1000;
    }

    clock_t start = clock();
    reverseArrayBlock(h_out, h_in, size);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_248_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_out[i]);
            }
            fclose(fp);
        }
    }

    free(h_in);
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_out);

    return 0;
}
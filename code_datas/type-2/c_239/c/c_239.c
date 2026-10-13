#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int DIM = 15000;
    int* h_in = (int*)malloc(DIM * DIM * sizeof(int));
    int* h_out = (int*)malloc(DIM * sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < DIM * DIM; ++i) {
        h_in[i] = i + 1;
    }

    clock_t start = clock();

    for (int i = 0; i < DIM; ++i) {
        int sum = 0;
        for (int j = 0; j < DIM; j++) {
            sum += h_in[i * DIM + j];
        }
        h_out[i] = sum;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_239_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_out[i]);
            }
            fclose(fp);
        }
    }


    free(h_in);
    free(h_out);

    return 0;
}
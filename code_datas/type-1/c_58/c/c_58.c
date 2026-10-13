#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void incKernel(int *g_out, int *g_in, int N, int inner_reps) {
    for (int idx = 0; idx < N; idx++) {
        for (int i = 0; i < inner_reps; ++i) {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}

int main(int argc, char *argv[]) {
    const int N = 100000;
    const int inner_reps = 100000;
    int* h_in = (int*)malloc(N * sizeof(int));
    int* h_out = (int*)malloc(N * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    clock_t start = clock();

    incKernel(h_out, h_in, N, inner_reps);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "c_58_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_out[%d]: %d\n", i, h_out[i]);
            }
            fclose(fp);
        }
    }


    free(h_in);
    free(h_out);

    return 0;
}
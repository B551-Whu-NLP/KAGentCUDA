#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_vec_scalaire(int* vec, int* res, int a, long N) {
    for (long i = 0; i < N; i++) {
        res[i] = vec[i] + a;
    }
}

int main(int argc, char* argv[]) {
    long N = 100000000; // 增加100倍
    int* h_vec;
    int* h_res;
    h_vec = (int*)malloc(N * sizeof(int));
    h_res = (int*)malloc(N * sizeof(int));

    for (long i = 0; i < N; ++i) {
        h_vec[i] = i;
    }

    clock_t start = clock();

    add_vec_scalaire(h_vec, h_res, 5, N);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_231_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (long i = 0; i < 1000; ++i) {
                fprintf(f, "%d ", h_res[i]);
            }
            fclose(f);
        }
    }

    free(h_vec);
    free(h_res);

    printf("Time: %.2f\n", time_taken);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void roundOff(float* mat, int N, int M, double time_taken) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (mat[i * N + j] >= 0)
                mat[i * N + j] = (int)(mat[i * N + j] + 0.5);
            else
                mat[i * N + j] = (int)(mat[i * N + j] - 0.5);
        }
        printf("Time: %.2f(ms)\n", time_taken);
    }
}

int main(int argc, char* argv[]) {
    const int N = 1000;
    const int M = 1000;
    float* h_mat = (float*)malloc(N * M * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);

    for (int i = 0; i < N * M; ++i) {
        h_mat[i] = rand() % 1000 / 10.0;
    }

    clock_t start = clock();

    // 增加计算量以延长运行时间
    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < N * M; i++) {
            h_mat[i] += 1.0f;
        }
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    roundOff(h_mat, N, M, time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_289_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    fprintf(f, "%.1f\t", h_mat[i * N + j]);
                }
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }

    free(h_mat);

    return 0;
}
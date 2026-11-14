#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void fill_idx(int N, int* device_input, int* device_output) {
    int idx;
    for (idx = 0; idx + 1 < N; idx++) {
        if (device_input[idx] + 1 == device_input[idx + 1]) {
            device_output[device_input[idx]] = idx;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 10000000;
    int* h_input = (int*)malloc(N * sizeof(int));
    int* h_output = (int*)calloc(N, sizeof(int));

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() % N;
    }

    clock_t start = clock();
    fill_idx(N, h_input, h_output);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_245_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_output[i]);
            }
            fclose(fp);
        }
    }

    free(h_input);
    free(h_output);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void shiftIndices(long* vec_out, const long by, const long imageSize, const long N) {
    for (long idx = 0; idx < N; idx++) {
        vec_out[idx] = (imageSize + ((idx - N / 2 + by) % imageSize)) % imageSize;
    }
}

int main(int argc, char* argv[]) {
    long N = 10000000;
    long imageSize = 800000;
    long shiftAmount = 3000;
    long* h_vec_out = (long*)malloc(N * sizeof(long));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (long i = 0; i < N; i++) {
        h_vec_out[i] = rand() % 100;
    }

    clock_t start = clock();
    shiftIndices(h_vec_out, shiftAmount, imageSize, N);
    clock_t end = clock();
    
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", argv[0]);
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (long i = 0; i < 1000; ++i) {
                fprintf(file, "%ld ", h_vec_out[i]);
            }
            fclose(file);
        }
    }

    free(h_vec_out);
    printf("Time: %.2f(ms)\n", timeElapsed);
    return 0;
}
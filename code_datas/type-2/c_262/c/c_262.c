#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sum_backward(float* db, float* dout, int r, int c) {
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            db[j] += dout[i * c + j];
        }
    }
}

int main(int argc, char* argv[]) {
    int r = 10000;
    int c = 10000;
    float* h_db = (float*)malloc(c * sizeof(float));
    float* h_dout = (float*)malloc(r * c * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < c; ++i) {
        h_db[i] = 0.0;
    }
    for (int i = 0; i < r * c; ++i) {
        h_dout[i] = i + 1;
    }

    clock_t start_time = clock();

    sum_backward(h_db, h_dout, r, c);

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_262_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_db[i]);
            }
            fclose(fp);
        }
    }

    free(h_db);
    free(h_dout);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sum_backward(float* db, float* sum, int r_sum, int c) {
    for (unsigned int j = 0; j < c; j++) {
        float temp = 0;

        for (int i = 0; i < r_sum; i++) {
            temp += sum[i * c + j];
        }

        db[j] = temp;
    }
}

int main(int argc, char* argv[]) {
    const int r_sum = 1000;
    const int c = 500;

    float* h_db = (float*)malloc(c * sizeof(float));
    float* h_sum = (float*)malloc(r_sum * c * sizeof(float));

    for (int i = 0; i < r_sum * c; ++i) {
        h_sum[i] = (float)(i);
    }

    clock_t start = clock();
    sum_backward(h_db, h_sum, r_sum, c);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_69_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "h_db[%d]: %f\n", i, h_db[i]);
            }
            fclose(file);
        }
    }


    free(h_db);
    free(h_sum);

    return 0;
}
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void Softmax_seg(float* x, const int c, const int size) {
    float temp1, temp2;

    for (int i = 0; i < size; i++) {
        temp1 = 0.;
        temp2 = 0.;

        for (int j = 0; j < c; j++) {
            temp1 = fmaxf(x[j * size + i], temp1);
        }

        for (int j = 0; j < c; j++) {
            x[j * size + i] = expf(x[j * size + i] - temp1);
            temp2 += x[j * size + i];
        }

        for (int j = 0; j < c; j++) {
            x[j * size + i] /= temp2;
        }
    }
}

int main(int argc, char* argv[]) {
    int c = 5000;
    int size = 10000;
    int total_elements = c * size;

    float* h_x = (float*)malloc(total_elements * sizeof(float));
    for (int i = 0; i < total_elements; i++) {
        h_x[i] = (float)(i % 100);
    }

    clock_t start_time = clock();

    Softmax_seg(h_x, c, size);

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_337_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 100; i++) {
                fprintf(file, "Element %d: ", i);
                for (int j = 0; j < 100; j++) {
                    fprintf(file, "%f ", h_x[j * size + i]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_x);
    return 0;
}
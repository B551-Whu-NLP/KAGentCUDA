#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void grad_x(const float* u, float* grad, long depth, long rows, long cols) {
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long size2d = rows * cols;
                unsigned long long idx = z * size2d + y * cols + x;

                if (x > 0) {
                    grad[idx] = u[idx] - u[idx - 1];
                }
                else {
                    grad[idx] = 3.0f;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    long depth = 128;
    long rows = 128;
    long cols = 128;

    float* h_u = (float*)malloc(depth * rows * cols * sizeof(float));
    float* h_grad = (float*)malloc(depth * rows * cols * sizeof(float));

    for (long i = 0; i < depth * rows * cols; ++i) {
        h_u[i] = (float)i;
        h_grad[i] = 6.0f;
    }

    clock_t start = clock();

    grad_x(h_u, h_grad, depth, rows, cols);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_139_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (long z = 0; z < depth; ++z) {
                for (long y = 0; y < rows; ++y) {
                    for (long x = 0; x < cols; ++x) {
                        long idx = z * rows * cols + y * cols + x;
                        fprintf(fp, "%f ", h_grad[idx]);
                    }
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }

    printf("Time: %.2fms\n", time_taken * 1000);

    free(h_u);
    free(h_grad);
    return 0;
}
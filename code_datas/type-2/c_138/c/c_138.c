#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void grad_y(const float* u, float* grad, long depth, long rows, long cols, int verbose, const char* file_name) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) { // 增加迭代次数
        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows; y++) {
                for (int z = 0; z < depth; z++) {
                    unsigned long size2d = rows * cols;
                    unsigned long long idx = z * size2d + y * cols + x;
                    float uidx = u[idx];

                    if (y > 0) {
                        grad[idx] = (uidx - u[z * size2d + (y - 1) * cols + x]);
                    }
                    else {
                        grad[idx] = 0.0f;
                    }
                }
            }
        }
    }

    double time_taken = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", file_name);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (long z = 0; z < depth; ++z) {
                fprintf(fp, "Depth slice %ld:\n", z);
                for (long y = 0; y < rows; ++y) {
                    for (long x = 0; x < cols; ++x) {
                        long idx = z * rows * cols + y * cols + x;
                        fprintf(fp, "%.6f ", grad[idx]);
                    }
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    long depth = 128;   // 增大矩阵维度
    long rows = 128;
    long cols = 128;

    int verbose = 0;
    const char* file_name = "c_138";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    float* h_u = (float*)malloc(depth * rows * cols * sizeof(float));
    float* h_grad = (float*)malloc(depth * rows * cols * sizeof(float));

    for (long i = 0; i < depth * rows * cols; ++i) {
        h_u[i] = (float)(2);
    }

    grad_y(h_u, h_grad, depth, rows, cols, verbose, file_name);

    free(h_u);
    free(h_grad);
    return 0;
}
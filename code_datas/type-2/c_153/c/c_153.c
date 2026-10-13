#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void l2normalize(float* x, float* dx, int batch, int filters, int spatial) {
    clock_t start = clock();

    int b, f, i;

    for (b = 0; b < batch; ++b) {
        for (i = 0; i < spatial; ++i) {
            float sum = 0;

            for (f = 0; f < filters; ++f) {
                int index = b * filters * spatial + f * spatial + i;
                sum += x[index] * x[index];
            }

            sum = sqrtf(sum);

            for (f = 0; f < filters; ++f) {
                int index = b * filters * spatial + f * spatial + i;
                x[index] /= sum;
                dx[index] = (1.0f - x[index]) / sum;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char* argv[]) {
    int batch = 100;
    int filters = 200;
    int spatial = 3000;

    float* x = (float*)malloc(batch * filters * spatial * sizeof(float));
    float* dx = (float*)malloc(batch * filters * spatial * sizeof(float));

    for (int i = 0; i < batch * filters * spatial; i++) {
        x[i] = i % 100 + 1;
    }

    l2normalize(x, dx, batch, filters, spatial);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_153_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < batch * filters ; i++) {
                fprintf(file, "%f ", x[i]);
            }
            fprintf(file, "\n");
            for (int i = 0; i < batch * filters; i++) {
                fprintf(file, "%f ", dx[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    } 

    free(x);
    free(dx);
    return 0;
}
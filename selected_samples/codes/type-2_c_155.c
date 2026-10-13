#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float* add, int w2, int h2, int c2, float* out) {
    clock_t start = clock();

    for (int id = 0; id < size; id++) {
        int i = id % minw;
        int j = (id / minw) % minh;
        int k = (id / (minw * minh)) % minc;
        int b = id / (minw * minh * minc) % batch;

        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

        out[out_index] += add[add_index];
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char** argv) {
    int size = 1000000;
    int minw = 128;
    int minh = 128;
    int minc = 128;
    int stride = 2;
    int sample = 3;
    int batch = 8;
    int w1 = 64;
    int h1 = 64;
    int c1 = 64;
    int w2 = 128;
    int h2 = 128;
    int c2 = 128;

    // Corrected memory allocation sizes
    long long add_size = (long long)minw * minh * minc * batch;
    long long out_size = (long long)sample * w2 * h2 * c2 * batch;

    float* add = (float*)malloc(add_size * sizeof(float));
    float* out = (float*)malloc(out_size * sizeof(float));

    if (add == NULL || out == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < add_size; i++) {
        add[i] = i + 1;
    }

    for (int i = 0; i < out_size; i++) {
        out[i] = i + 2;
    }

    shortcut_kernel(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, out);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_155_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%f ", out[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(add);
    free(out);
    return 0;
}
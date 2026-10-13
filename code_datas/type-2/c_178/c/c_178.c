#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void opLadj2(float* vec, float* vec1, float* vec2, float* vec3, long depth, long rows, long cols, int verbose, const char* file_name) {
    clock_t start = clock();

    for (int iter = 0; iter < 10000; iter++) {
        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows; y++) {
                for (int z = 0; z < depth; z++) {
                    unsigned long long i = z * rows * cols + y * cols + x;
                    unsigned long long j = z * rows * cols + y * cols;
                    unsigned long size2d = z * rows * cols + cols * rows;
                    unsigned long size3d = depth * rows * cols + rows * cols + cols;

                    if (i + cols + 1 >= size3d) return;

                    vec[i + 1] = vec1[i + 1] + 0.25 * (vec2[i + 1] + vec2[i] + vec2[i + cols + 1] + vec2[i + cols]) + 0.5 * (vec3[i + 1] + vec3[i + cols + 1]);

                    if (j + cols >= size2d) return;

                    vec[j] = vec1[j] + (vec2[j] + vec2[j + cols]) / 4 + (vec3[j] + vec3[j + cols]) / 2;
                }
            }
        }
    }

    double timeElapsed = ((double)(clock() - start) * 1000) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", file_name);
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < depth * rows * cols; ++i) {
                fprintf(fp, "%.2f ", vec[i]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char** argv) {
    const long depth = 128;
    const long rows = 256;
    const long cols = 256;

    int verbose = 0;
    const char* file_name = "c_178";

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* vec, * vec1, * vec2, * vec3;
    vec = (float*)malloc(depth * rows * cols * sizeof(float));
    vec1 = (float*)malloc(depth * rows * cols * sizeof(float));
    vec2 = (float*)malloc(depth * rows * cols * sizeof(float));
    vec3 = (float*)malloc(depth * rows * cols * sizeof(float));

    for (long i = 0; i < depth * rows * cols; ++i) {
        vec[i] = 0.0f;
        vec1[i] = (float)(i % 100);
        vec2[i] = (float)((i + 1) % 100);
        vec3[i] = (float)((i + 2) % 100);
    }

    opLadj2(vec, vec1, vec2, vec3, depth, rows, cols, verbose, file_name);

    free(vec);
    free(vec1);
    free(vec2);
    free(vec3);

    return 0;
}
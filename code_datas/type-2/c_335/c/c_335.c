#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void opLadj1(float* vec, float* vec1, float* vec2, float* vec3, long depth, long rows, long cols, int verbose) {
    clock_t start = clock();

    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long long i = z * rows * cols + y * cols + x;
                unsigned long long j = z * rows * cols + x;
                unsigned long size2d = cols;
                unsigned long size3d = depth * rows * cols + rows * cols + cols;

                if (i + cols + 1 >= size3d)
                    return;

                vec[i + cols] = vec1[i + cols] + 0.25 * (vec2[i + cols] + vec2[i] + vec2[i + cols + 1] + vec2[i + 1]) +
                    0.5 * (vec3[i + cols] + vec3[i + cols + 1]);

                if (j + 1 >= size2d)
                    return;

                vec[j] = vec1[j] + (vec2[j] + vec2[j + 1]) / 4 + (vec3[j] + vec3[j + 1]) / 2;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_335_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < depth; i += 4) { 
                for (int j = 0; j < rows; j += 128) { 
                    for (int k = 0; k < cols; k += 128) { 
                        fprintf(fp, "%f\n", vec[i * rows * cols + j * cols + k]);
                    }
                }
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    const long depth = 16;
    const long rows = 128;
    const long cols = 128;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* vec, * vec1, * vec2, * vec3;
    vec = (float*)malloc(depth * rows * cols * sizeof(float));
    vec1 = (float*)malloc(depth * rows * cols * sizeof(float));
    vec2 = (float*)malloc(depth * rows * cols * sizeof(float));
    vec3 = (float*)malloc(depth * rows * cols * sizeof(float));

    for (int i = 0; i < depth * rows * cols; ++i) {
        vec[i] = (float)(i % 100);
        vec1[i] = (float)(i % 100);
        vec2[i] = (float)(i % 100);
        vec3[i] = (float)(i % 100);
    }

    opLadj1(vec, vec1, vec2, vec3, depth, rows, cols, verbose);

    free(vec);
    free(vec1);
    free(vec2);
    free(vec3);

    return 0;
}
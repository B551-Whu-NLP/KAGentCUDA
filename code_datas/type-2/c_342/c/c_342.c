#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void opL21(float* vec, float* vec1, long depth, long rows, long cols) {
    clock_t start = clock();
    
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long long i = z * rows * cols + y * cols + x;
                unsigned long long j = z * rows * cols + x;
                unsigned long size2d = cols;
                unsigned long size3d = depth * rows * cols + rows * cols + cols;
                if (i + cols + 1 >= size3d || j + 1 >= size2d) {
                    return;
                }
                vec[i + cols] = 0.25 * (vec1[i + 1] + vec1[i] + vec1[i + cols + 1] + vec1[i + cols]);
                vec[j] = (vec1[j] + vec1[j + 1]) / 4;
            }
        }
    }
    
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    const long depth = 16;
    const long rows = 32;
    const long cols = 16;

    float* vec, * vec1;
    vec = (float*)malloc(depth * rows * cols * sizeof(float));
    vec1 = (float*)malloc(depth * rows * cols * sizeof(float));

    for (long i = 0; i < depth * rows * cols; ++i) {
        vec1[i] = (float)(i * 2);
        vec[i] = 0;
    }

    opL21(vec, vec1, depth, rows, cols);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_342_output.txt", "w");
        if (fp != NULL) {
            for (long i = 0; i < depth; i++) {
                for (long j = 0; j < rows; j++) {
                    for (long k = 0; k < cols; k++) {
                        long index = i * rows * cols + j * cols + k;
                        fprintf(fp, "vec[%ld] = %.2f\n", index, vec[index]);
                    }
                }
            }
            fclose(fp);
        }
    } 

    free(vec);
    free(vec1);
    return 0;
}
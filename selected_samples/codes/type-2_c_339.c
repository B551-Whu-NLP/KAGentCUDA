#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Ring_kernel(float* A, float* BP, int* corrAB, float* M, int ring, int c, int h, int w) {
    int ringSize = 2 * ring + 1;
    int ringPatch = ringSize * ringSize;
    int size = h * w;

    clock_t start = clock();

    for (int y1 = 0; y1 < h; y1++) {
        for (int x1 = 0; x1 < w; x1++) {
            int id = y1 * w + x1;
            int x2 = corrAB[2 * id + 0];
            int y2 = corrAB[2 * id + 1];

            for (int dx = -ring; dx <= ring; dx++) {
                for (int dy = -ring; dy <= ring; dy++) {
                    int pIdx = (dy + ring) * ringSize + (dx + ring);
                    int _x2 = x2 + dx, _y2 = y2 + dy;

                    if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h) {
                        for (int dc = 0; dc < c; dc++) {
                            M[(dc * size + y1 * w) * ringPatch + pIdx * w + x1] = BP[dc * size + _y2 * w + _x2];
                        }
                    }
                }
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);

    return;
}

int main(int argc, char *argv[]) {
    const int h = 64;
    const int w = 64;
    const int c = 3;
    const int ring = 2;
    float* A, * BP, * M;
    int* corrAB;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    A = (float*)malloc(h * w * sizeof(float));
    BP = (float*)malloc(c * h * w * sizeof(float));
    M = (float*)malloc(c * h * w * (2 * ring + 1) * (2 * ring + 1) * sizeof(float));
    corrAB = (int*)malloc(2 * h * w * sizeof(int));

    for (int i = 0; i < h * w; ++i) {
        A[i] = (float)(i);
        corrAB[2 * i] = i % w;
        corrAB[2 * i + 1] = i / w;
    }

    for (int i = 0; i < c * h * w; ++i) {
        BP[i] = (float)(i);
    }

    Ring_kernel(A, BP, corrAB, M, ring, c, h, w);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_339_output.txt");

        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int dc = 0; dc < c; dc++) {
                fprintf(fp, "Channel %d:\n", dc);
                for (int i = 0; i < h * w; i++) {
                    for (int dy = -ring; dy <= ring; dy++) {
                        for (int dx = -ring; dx <= ring; dx++) {
                            int pIdx = (dy + ring) * (2 * ring + 1) + (dx + ring);
                            fprintf(fp, "%.2f ", M[(dc * h * w) * (2 * ring + 1) * (2 * ring + 1) + pIdx * w + (i % w)]);
                        }
                        fprintf(fp, "\n");
                    }
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }

    free(A);
    free(BP);
    free(M);
    free(corrAB);

    return 0;
}
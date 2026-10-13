#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convolute(float* dData, float* hData, int height, int width, float* mask, int masksize, int verbose) {
    clock_t start = clock();

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int S = (masksize - 1) / 2;
            float sum = 0;
            int pixPos = row * width + col;
            dData[pixPos] = 0.0;

            for (int maskrow = -S; maskrow <= S; maskrow++) {
                for (int maskcol = -S; maskcol <= S; maskcol++) {
                    int pixP = (row + maskrow) * width + (col + maskcol);
                    int maskP = (maskrow + S) * masksize + (maskcol + S);

                    if (pixP < height * width && pixP >= 0 && maskP < masksize * masksize) {
                        sum += mask[maskP] * hData[pixP];
                    }
                }
            }

            dData[pixPos] = sum;

            if (dData[pixPos] < 0) {
                dData[pixPos] = 0;
            }
            else if (dData[pixPos] > 1) {
                dData[pixPos] = 1;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_179_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int row = 0; row < 1000; row++) {
                for (int col = 0; col < 1000; col++) {
                    fprintf(fp, "%.2f ", dData[row * width + col]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    const int height = 2048;
    const int width = 2048;
    const int masksize = 9;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* dData, * hData, * mask;

    hData = (float*)malloc(height * width * sizeof(float));
    dData = (float*)malloc(height * width * sizeof(float));
    mask = (float*)malloc(masksize * masksize * sizeof(float));

    srand(0);
    for (int i = 0; i < height * width; ++i) {
        hData[i] = (float)(rand() % 256) / 255.0f;
    }

    for (int i = 0; i < masksize * masksize; ++i) {
        mask[i] = 1.0f / (masksize * masksize);
    }

    convolute(dData, hData, height, width, mask, masksize, verbose);

    free(hData);
    free(dData);
    free(mask);

    return 0;
}
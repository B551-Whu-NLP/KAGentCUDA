#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void subtractMean(double* images, const double* meanImage, int imageNum, int pixelNum) {
    for (int col = 0; col < pixelNum; col++) {
        for (int row = 0; row < imageNum; ++row) {
            images[row * pixelNum + col] -= meanImage[col];
            if (images[row * pixelNum + col] < 0.0) {
                images[row * pixelNum + col] = 0.0;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int imageNum = 8192;
    int pixelNum = 16384;
    double* h_images = (double*)malloc(imageNum * pixelNum * sizeof(double));
    double* h_meanImage = (double*)malloc(pixelNum * sizeof(double));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < pixelNum; ++i) {
        h_meanImage[i] = -i;
    }
    for (int i = 0; i < imageNum * pixelNum; ++i) {
        h_images[i] = 0.0;
    }

    clock_t start = clock();
    subtractMean(h_images, h_meanImage, imageNum, pixelNum);
    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        FILE *fp = fopen("c_102_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    fprintf(fp, "%.2f ", h_images[i * pixelNum + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_images);
    free(h_meanImage);
    return 0;
}
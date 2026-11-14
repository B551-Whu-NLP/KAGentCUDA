#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double getMeanImage(const double* images, double* meanImage, int imageNum, int pixelNum) {
    clock_t start = clock();

    for (int col = 0; col < pixelNum; col++) {
        meanImage[col] = 0.0;
        for (int row = 0; row < imageNum; ++row) {
            meanImage[col] += images[row * pixelNum + col];
        }
        meanImage[col] /= imageNum;
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char** argv) {
    const int imageNum = 10000;
    const int pixelNum = 10240;
    double* h_images = (double*)malloc(imageNum * pixelNum * sizeof(double));
    double* h_meanImage = (double*)malloc(pixelNum * sizeof(double));

    for (int i = 0; i < imageNum * pixelNum; ++i) {
        h_images[i] = i;
    }

    double timeElapsed = getMeanImage(h_images, h_meanImage, imageNum, pixelNum);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_284_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f\t", h_meanImage[i]);
            }
            fclose(file);
        }
    } 

    free(h_images);
    free(h_meanImage);

    printf("Time: %.2f(ms)\n", timeElapsed);
    return 0;
}
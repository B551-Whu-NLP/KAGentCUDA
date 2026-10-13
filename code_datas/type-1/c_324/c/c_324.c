#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void castImageToGrayScale(unsigned char *ucharImage, unsigned char *grayImage, int imageWidth, int imageHeight, int channels) {
    int w, h;
    for (w = 0; w < imageWidth; w++) {
        for (h = 0; h < imageHeight; h++) {
            int idx = imageWidth * h + w;
            unsigned char r = ucharImage[idx * channels];
            unsigned char g = ucharImage[idx * channels + 1];
            unsigned char b = ucharImage[idx * channels + 2];
            grayImage[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

int main(int argc, char *argv[]) {
    int imageWidth = 1000;
    int imageHeight = 1000;
    int channels = 3;

    unsigned char *ucharImage = (unsigned char *)malloc(imageWidth * imageHeight * channels * sizeof(unsigned char));
    unsigned char *grayImage = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));

    for (int i = 0; i < imageWidth * imageHeight * channels; i++) {
        ucharImage[i] = (i + 1) % 256;
    }

    clock_t start = clock();
    for(int iter = 0; iter < 100; iter++) {
        castImageToGrayScale(ucharImage, grayImage, imageWidth, imageHeight, channels);
    }
    clock_t end = clock();
    double timeElapsed = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_324_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < imageHeight; i++) {
                for (int j = 0; j < imageWidth; j++) {
                    fprintf(fp, "%3u ", grayImage[imageWidth * i + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

 
    printf("Time: %.2f(ms)\n", timeElapsed);   free(ucharImage);
    free(grayImage);

    return 0;
}
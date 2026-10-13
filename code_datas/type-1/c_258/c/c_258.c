#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void castImageToUchar(float* deviceInputImageData, unsigned char* ucharImage,
    int imageWidth, int imageHeight, int channels, int pixelSize) {
    int w;
    for (w = 0; w < pixelSize; w++) {
        ucharImage[w] = (unsigned char)(255 * deviceInputImageData[w]);
    }
}

int main(int argc, char* argv[]) {
    int imageWidth = 1000;
    int imageHeight = 1000;
    int channels = 3;
    int pixelSize = channels * imageWidth * imageHeight;
    float* h_deviceInputImageData = (float*)malloc(pixelSize * sizeof(float));
    unsigned char* h_ucharImage = (unsigned char*)malloc(pixelSize * sizeof(unsigned char));
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < pixelSize; ++i) {
        h_deviceInputImageData[i] = (float)(i % 256) / 255.0f;
    }

    clock_t start = clock();

    for(int iter = 0; iter < 1000; iter++) {
        castImageToUchar(h_deviceInputImageData, h_ucharImage, imageWidth, imageHeight, channels, pixelSize);
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        sprintf(outputFileName, "%s_output.txt", "c_258");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_ucharImage[i]);
            }
            fclose(file);
        }
    } 

    free(h_deviceInputImageData);
    free(h_ucharImage);

    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    return 0;
}
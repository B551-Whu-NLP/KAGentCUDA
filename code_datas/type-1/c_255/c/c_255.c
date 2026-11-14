#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void castImageTofloat(float* deviceOutputImageData, unsigned char* ucharImage,
    int imageWidth, int imageHeight, int channels, int pixelSize) {
    for (int w = 0; w < pixelSize; w++) {
        deviceOutputImageData[w] = (float)(ucharImage[w] / 255.0);
    }
}

int main(int argc, char** argv) {
    int imageWidth = 1000;
    int imageHeight = 1000;
    int channels = 3;
    int pixelSize = channels * imageWidth * imageHeight;
    float* h_deviceOutputImageData = (float*)malloc(pixelSize * sizeof(float));
    unsigned char* h_ucharImage = (unsigned char*)malloc(pixelSize * sizeof(unsigned char));

    for (int i = 0; i < pixelSize; ++i) {
        h_ucharImage[i] = (char)(i % 256);
    }

    clock_t start = clock();

    // Repeat the operation multiple times to increase execution time
    for (int rep = 0; rep < 1000; rep++) {
        castImageTofloat(h_deviceOutputImageData, h_ucharImage, imageWidth, imageHeight, channels, pixelSize);
    }

    clock_t end = clock();
    
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_255_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%f ", h_deviceOutputImageData[i]);
            }
            fclose(file);
        }
    }

    free(h_deviceOutputImageData);
    free(h_ucharImage);
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    return 0;
}
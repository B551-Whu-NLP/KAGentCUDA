#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void equalization(float* cdf, float* mincdf, unsigned char* ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize) {
    int idx;
    for (idx = 0; idx < pixelSize; idx++) {
        unsigned char val = ucharImage[idx];
        float data = 255 * (cdf[val] - mincdf[0]) / (1 - mincdf[0]);
        if (data < 0.0f) {
            data = 0.0f;
        }
        else if (data > 255.0f) {
            data = 255.0f;
        }
        ucharImage[idx] = (unsigned char)data;
    }
}

int main(int argc, char *argv[]) {
    int imageWidth = 1000;
    int imageHeight = 1000;
    int channels = 3;
    int pixelSize = imageWidth * imageHeight * channels;

    float *cdf = (float *)malloc(256 * sizeof(float));
    float *mincdf = (float *)malloc(1 * sizeof(float));
    unsigned char *ucharImage = (unsigned char *)malloc(pixelSize * sizeof(unsigned char));

    for (int i = 0; i < 256; i++) {
        cdf[i] = i / 255.0; 
    }

    for (int i = 0; i < 1; i++) {
        mincdf[i] = 0.1; 
    }

    for (int i = 0; i < pixelSize; i++) {
        ucharImage[i] = i % 256; 
    }

    clock_t start = clock();
    for (int iter = 0; iter < 1000; iter++) {
        equalization(cdf, mincdf, ucharImage, imageWidth, imageHeight, channels, pixelSize);
    }
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_309_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            fprintf(fp, "Equalized Image:\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%u ", ucharImage[i]);
                if ((i + 1) % channels == 0) {
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    free(cdf);
    free(mincdf);
    free(ucharImage);

    return 0;
}
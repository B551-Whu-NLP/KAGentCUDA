#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void colorConvert(unsigned char* grayImage, unsigned char* colorImage, int rows, int columns, int verbose) {
    clock_t start = clock();

    for (int column = 0; column < columns; column++) {
        for (int row = 0; row < rows; row++) {
            int offset = column + (columns * row);
            unsigned char grayValue = 0.07 * colorImage[offset * 3] + 0.71 * colorImage[offset * 3 + 1] + 0.21 * colorImage[offset * 3 + 2];
            grayImage[offset] = grayValue;
        }
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char outputFileName[256];
        snprintf(outputFileName, sizeof(outputFileName), "c_119_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", grayImage[i]);
            }
            fclose(file);
        }
    }
}

int main(int argc, char* argv[]) {
    int rows = 5000;
    int columns = 5000;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    unsigned char* h_grayImage = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    unsigned char* h_colorImage = (unsigned char*)malloc(rows * columns * 3 * sizeof(unsigned char));

    for (int i = 0; i < rows * columns * 3; i++) {
        h_colorImage[i] = (unsigned char)(i % 256);
    }

    colorConvert(h_grayImage, h_colorImage, rows, columns, verbose);

    free(h_grayImage);
    free(h_colorImage);
    return 0;
}
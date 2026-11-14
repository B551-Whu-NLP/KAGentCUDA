#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void histogrammPrimitive(unsigned int* histogrammVector, unsigned char* grayImage, int rows, int columns) {
    int column;
    int row;
    for (column = 0; column < columns; column++) {
        for (row = 0; row < rows; row++) {
            int offset = (column)+(columns * row);
            unsigned char grayValue = grayImage[offset];
            histogrammVector[grayValue]++;
        }
    }
}

int main(int argc, char *argv[]) {
    const int rows = 10000;
    const int columns = 10000;
    const int histogramSize = 256;
    unsigned char* h_grayImage = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    unsigned int* h_histogramVector = (unsigned int*)calloc(histogramSize, sizeof(unsigned int));

    for (int i = 0; i < rows * columns; ++i) {
        h_grayImage[i] = i % 256;
    }

    clock_t start = clock();
    histogrammPrimitive(h_histogramVector, h_grayImage, rows, columns);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_292_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Value %d: %u\n", i, h_histogramVector[i]);
            }
            fclose(fp);
        }
    }

    free(h_grayImage);
    free(h_histogramVector);

    return 0;
}
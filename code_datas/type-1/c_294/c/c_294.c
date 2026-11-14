#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convertInstanceToLabel(unsigned short* d_outputLabel, const unsigned char* d_inputInstance,
    const unsigned short* d_instanceToLabel, unsigned int width,
    unsigned int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            d_outputLabel[y * width + x] = d_instanceToLabel[d_inputInstance[y * width + x]];
        }
    }
}

int main(int argc, char* argv[]) {
    const unsigned int width = 5000;
    const unsigned int height = 5000;
    unsigned char* h_inputInstance = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned short* h_instanceToLabel = (unsigned short*)malloc(256 * sizeof(unsigned short));
    unsigned short* h_outputLabel = (unsigned short*)malloc(width * height * sizeof(unsigned short));

    srand(42);
    for (unsigned int i = 0; i < width * height; ++i) {
        h_inputInstance[i] = (unsigned char)(rand() % 256);
    }
    for (int i = 0; i < 256; ++i) {
        h_instanceToLabel[i] = (unsigned short)(i);
    }

    clock_t start = clock();
    convertInstanceToLabel(h_outputLabel, h_inputInstance, h_instanceToLabel, width, height);
    clock_t end = clock();

    double time_taken = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_294_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(file, "Element %u: %u\n", i, h_outputLabel[i]);
            }
            fclose(file);
        }
    }

    free(h_inputInstance);
    free(h_instanceToLabel);
    free(h_outputLabel);

    return 0;
}
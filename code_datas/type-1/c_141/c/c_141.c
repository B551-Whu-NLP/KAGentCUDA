#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void apply_grayscale(const unsigned char* image, unsigned char* grayimg, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const unsigned char R = image[(y * width + x) * 3 + 0];
            const unsigned char G = image[(y * width + x) * 3 + 1];
            const unsigned char B = image[(y * width + x) * 3 + 2];
            unsigned char gray = (307 * R + 604 * G + 113 * B) >> 10;
            grayimg[y * width + x] = gray;
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 1024;
    int height = 1024;

    unsigned char* h_image = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
    unsigned char* h_grayimg = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    for (int i = 0; i < width * height * 3; ++i) {
        h_image[i] = i % 256;
    }

    clock_t start_time = clock();

    // Repeat the operation multiple times to increase runtime
    for (int iter = 0; iter < 100; iter++) {
        apply_grayscale(h_image, h_grayimg, width, height);
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_141_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    fprintf(fp, "%d ", h_grayimg[y * width + x]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

   
    printf("Time: %.2f(ms)\n", elapsed_time); free(h_image);
    free(h_grayimg);
    return 0;
}
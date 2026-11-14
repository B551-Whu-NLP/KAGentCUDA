#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void convertKinectDisparityInPlace(float* d_disparity, int pitch, int width, int height, float depth_scale) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float* d_in = (float*)((char*)d_disparity + y * pitch) + x;
            *d_in = (*d_in == 0.0f) ? 1 : (-depth_scale / *d_in);
        }
    }
}

int main(int argc, char** argv) {
    int width = 4096;
    int height = 3072;
    int pitch = width * sizeof(float);
    float depth_scale = 0.01f;
    int verbose = 0;
    clock_t start = clock();

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_disparity = (float*)malloc(width * height * sizeof(float));

    for (int i = 0; i < width * height; i++) {
        h_disparity[i] = (i % 100) + 1;
    }

    convertKinectDisparityInPlace(h_disparity, pitch, width, height, depth_scale);

    if (verbose) {
        FILE* file = fopen("c_106_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%f ", h_disparity[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(h_disparity);
    printf("Time: %.2f(ms)\n", ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000);
    return 0;
}
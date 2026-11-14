#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void convertKinectDisparityToRegularDisparity(float* regularDisparity, int regularDisparityPitch, const float* kinectDisparity, int kinectDisparityPitch, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float d_in = *((float*)((char*)kinectDisparity + y * kinectDisparityPitch) + x);
            float d_out = (d_in == 0.0f) ? 1 : -d_in;
            *((float*)((char*)regularDisparity + y * regularDisparityPitch) + x) = d_out;
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 1000;
    int height = 1000;

    float* kinectDisparity = (float*)malloc(height * width * sizeof(float));
    float* regularDisparity = (float*)malloc(height * width * sizeof(float));

    srand(42);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            kinectDisparity[y * width + x] = (float)(rand() % 1000) / 100.0f; 
        }
    }

    clock_t start = clock();

    for (int i = 0; i < 100; i++) {
        convertKinectDisparityToRegularDisparity((float *)regularDisparity, width * sizeof(float), (float *)kinectDisparity, width * sizeof(float), width, height);
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_151_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    fprintf(fp, "%.2f ", regularDisparity[y * width + x]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
    free(kinectDisparity);
    free(regularDisparity);

    return 0;
}
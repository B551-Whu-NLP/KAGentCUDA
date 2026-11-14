#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void deInterleave(float* d_X_out, float* d_Y_out, char* d_XY_in, int pitch_out, int pitch_in, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float* data = (float*)(d_XY_in + y * pitch_in) + 2 * x;
            *((float*)((char*)d_X_out + y * pitch_out) + x) = data[0];
            *((float*)((char*)d_Y_out + y * pitch_out) + x) = data[1];
        }
    }
}

int main(int argc, char* argv[]) {
    const int width = 1000;
    const int height = 1000;
    const int pitch_out = width * sizeof(float);
    const int pitch_in = 2 * width * sizeof(float);

    float* h_X_out = (float*)malloc(width * height * sizeof(float));
    float* h_Y_out = (float*)malloc(width * height * sizeof(float));
    char* h_XY_in = (char*)malloc(2 * width * height * sizeof(float));

    for (int i = 0; i < 2 * width * height; i++) {
        ((float*)h_XY_in)[i] = (float)(i + 1);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 100; iter++) {
        deInterleave(h_X_out, h_Y_out, h_XY_in, pitch_out, pitch_in, width, height);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_331_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_X_out[i]);
                if ((i + 1) % width == 0) fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; i++) {  
                fprintf(fp, "%.2f ", h_Y_out[i]);
                if ((i + 1) % width == 0) fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_X_out);
    free(h_Y_out);
    free(h_XY_in);
    return 0;
}
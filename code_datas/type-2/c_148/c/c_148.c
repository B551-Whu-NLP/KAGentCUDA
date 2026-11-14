#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void normalizacion(float* image_c, long int image_size, int bands) {
    long int i, j;
    float norm_val = 0, aux = 0, pixel = 0;

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (i = 0; i < image_size; i++) {
            for (j = 0; j < bands; j++) {
                norm_val += image_c[j * image_size + i];
            }

            norm_val = 1.0 / (norm_val + 1.0e-16);

            for (j = 0; j < bands; j++) {
                pixel = image_c[j * image_size + i] * norm_val;
                image_c[j * image_size + i] = pixel;
                aux += pixel * pixel;  
            }
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_spent);
}

int main(int argc, char *argv[]) {
    int bands = 3;
    long int image_size = 1024 * 1024;
    long int total_size = bands * image_size;

    float* h_image = (float*)malloc(total_size * sizeof(float));

    for (long int i = 0; i < total_size; i++) {
        h_image[i] = (float)(i % 256);
    }

    normalizacion(h_image, image_size, bands);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[] = "c_148_output.txt";
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < bands; i++) {
                for (long int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.6f ", h_image[i * image_size + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_image);

    return 0;
}
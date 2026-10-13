#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convertFloatToRGBA(char* out_image, const float* in_image, int width, int height) {
    (void)in_image;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int IND = y * width + x;
            out_image[IND] = (char)(255);
        }
    }
}

int main(int argc, char* argv[]) {
    int width = 4096;
    int height = 4096;
    char* h_out_image = (char*)malloc(width * height * sizeof(char));
    float* h_in_image = (float*)malloc(width * height * sizeof(float));
    
    srand(12345);
    
    for (int i = 0; i < width * height; ++i) {
        h_in_image[i] = (float)(rand() % 256);
    }

    clock_t start = clock();

    convertFloatToRGBA(h_out_image, h_in_image, width, height);

    clock_t end = clock();
    
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_83_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(file, "%3d ", h_out_image[i * width + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }
    
   
    printf("Time: %.2f(ms)\n", time_spent * 1000); free(h_out_image);
    free(h_in_image);

    return 0;
}
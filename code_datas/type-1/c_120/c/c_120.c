#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void init_image_array(unsigned long long int* image, int pixels_per_image) {
    for (int my_pixel = 0; my_pixel < pixels_per_image * 4; my_pixel++) {
        image[my_pixel] = (unsigned long long int)(6);
    }
}

int main(int argc, char *argv[]) {
    int pixels_per_image = 1000000;
    unsigned long long int* image = (unsigned long long int*)malloc(pixels_per_image * 4 * sizeof(unsigned long long int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start_time = clock();

    for (int iter = 0; iter < 1000; iter++) {
        init_image_array(image, pixels_per_image);
    }

    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", elapsed_time * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_120_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%llu ", image[i]);
            }
            fclose(fp);
        }
    }

    free(image);

    return 0;
}